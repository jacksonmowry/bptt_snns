#!/usr/bin/env bash
# update_golden.sh
# Regenerate golden files in-place for one or all tests.
#
# Usage:
#   ./tests/update_golden.sh              # regenerate all tests
#   ./tests/update_golden.sh <test_name>  # regenerate one test
#
# If a golden network JSON already exists, shows a line-by-line diff
# and asks whether to accept the new version or keep the old.
#
# Tests live in tests/golden/<dir>/ with golden_test_N_<name>.cfg as config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_DIR="$SCRIPT_DIR/golden"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT_DIR/bin/bptt_learning"

# Build if needed
if [ ! -f "$BIN" ] || [ "$ROOT_DIR/src/bptt_learning.cpp" -nt "$BIN" ]; then
    echo "Building..."
    make -C "$ROOT_DIR" -j"$(nproc)" 2>&1 | tail -1
fi

# Filter stdout: keep only lines with essential metrics (epoch, final, CPU eval).
# Strips: "Neurons:", "Beginning training", device tables, OpenCL compile info,
# kernel timing report, etc.
filter_stdout() {
    local input="$1" output="$2"
    grep -E '^(Epoch:|Final |  \[CPU eval)' "$input" > "$output" || true
}

# Parse a .cfg file into space-separated CLI args
parse_cfg() {
    local cfg_file="$1"
    while IFS= read -r line; do
        line="${line%%#*}"       # strip comments
        line="${line// /}"       # strip spaces
        [[ -z "$line" ]] && continue
        local key="${line%%=*}"
        local val="${line#*=}"
        echo "--${key} ${val} "
    done < "$cfg_file"
}

# Show a line-by-line diff between old and new network JSON.
# Both are normalized (timestamps stripped) and pretty-printed.
show_net_diff() {
    local old_net="$1" new_net="$2"
    python3 - "$old_net" "$new_net" <<'PYEOF'
import json, sys, difflib

def normalize(json_path):
    with open(json_path) as f:
        obj = json.load(f)
    other = obj.setdefault("Associated_Data", {}).setdefault("other", {})
    for k in ("compile_time", "start_time", "git_commit"):
        other.pop(k, None)
    if "cli_args" in other and isinstance(other["cli_args"], list):
        args = list(other["cli_args"])
        while "--network_json_out" in args:
            idx = args.index("--network_json_out")
            args.pop(idx)
            if idx < len(args):
                args.pop(idx)
        other["cli_args"] = sorted(args)
    other.pop("network_json_out", None)
    return obj

a = normalize(sys.argv[1])
b = normalize(sys.argv[2])
sa = json.dumps(a, sort_keys=True, indent=2)
sb = json.dumps(b, sort_keys=True, indent=2)
diff = list(difflib.unified_diff(
    sa.splitlines(), sb.splitlines(),
    fromfile="old (golden)", tofile="new",
    lineterm=""
))
if diff:
    for line in diff:
        print(line)
PYEOF
}

# Run one test and save golden files
run_and_save() {
    local cfg_file="$1"
    local test_name
    test_name="$(basename "$cfg_file" .cfg)"
    local test_dir
    test_dir="$(dirname "$cfg_file")"

    local golden_out="$test_dir/${test_name}.out"
    local golden_net="$test_dir/${test_name}.net.json"
    local tmp_stdout
    local tmp_net
    tmp_stdout="$(mktemp)"
    tmp_net="$(mktemp)"

    local cli_args
    cli_args="$(parse_cfg "$cfg_file")"

    echo -n "Generating golden for ${test_name}... "

    if ! "$BIN" ${cli_args} --network_json_out "$tmp_net" > "$tmp_stdout" 2>&1; then
        echo "FAIL (command failed)"
        rm -f "$tmp_stdout" "$tmp_net"
        return 1
    fi

    # Always update stdout golden
    # Filter and save stdout golden
    filter_stdout "$tmp_stdout" "$golden_out"

    # Handle network JSON: diff if golden exists, prompt user
    if [ -f "$golden_net" ]; then
        diff_output="$(show_net_diff "$golden_net" "$tmp_net" 2>&1)" || true
        if [ -n "$diff_output" ]; then
            echo ""
            echo "--- Diff for ${test_name} network JSON ---"
            echo "$diff_output"
            echo "--- End diff ---"
            echo ""
            echo "Accept new version? [y/n] (y=accept, n=keep old)"
            read -r answer
            case "$answer" in
                [yY][eE][sS]|[yY])
                    cp "$tmp_net" "$golden_net"
                    echo "Accepted new version."
                    ;;
                *)
                    echo "Keeping old version."
                    ;;
            esac
        else
            echo "Network JSON unchanged."
        fi
    else
        cp "$tmp_net" "$golden_net"
        echo "saved"
    fi

    rm -f "$tmp_stdout" "$tmp_net"
}

# Decide which tests to run
if [ $# -ge 1 ]; then
    # Specific test requested — accepts full name or short name (auto-prefixes)
    test_name="$1"
    if [[ "$test_name" != golden_test_* ]]; then
        test_name="golden_test_${test_name}"
    fi
    # Search recursively for the config file
    cfg_file="$(find "$GOLDEN_DIR" -name "${test_name}.cfg" -type f 2>/dev/null | head -1)"
    if [ -z "$cfg_file" ]; then
        echo "Error: config not found for ${test_name}"
        exit 1
    fi
    run_and_save "$cfg_file"
else
    # All tests — find all .cfg files recursively
    cfgs=()
    while IFS= read -r -d '' f; do
        cfgs+=("$f")
    done < <(find "$GOLDEN_DIR" -name "golden_test_*.cfg" -type f -print0)

    if [ ${#cfgs[@]} -eq 0 ]; then
        echo "No golden tests found in $GOLDEN_DIR"
        exit 1
    fi
    for cfg_file in "${cfgs[@]}"; do
        run_and_save "$cfg_file"
    done
fi
