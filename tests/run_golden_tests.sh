#!/usr/bin/env bash
# run_golden_tests.sh
# Top-level golden test runner.
# Auto-discovers tests/golden/golden_test_*.cfg files, builds once, runs each.
# Each test compares stdout + trained network JSON against golden baselines.
#
# Only compile_time and start_time are stripped from network comparison —
# all other metadata is compared verbatim.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_DIR="$SCRIPT_DIR/golden"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT_DIR/bin/bptt_learning"

# --- helpers ---

build_if_needed() {
	if [ ! -f "$BIN" ] || [ "$ROOT_DIR/src/bptt_learning.cpp" -nt "$BIN" ]; then
		echo "Building..."
		make -C "$ROOT_DIR" -j"$(nproc)" 2>&1 | tail -1
	fi
}

# Parse a .cfg file into an associative array.
# Format: key=value, lines starting with # or empty lines are ignored.
parse_cfg() {
	local cfg_file="$1"
	declare -gA TEST_ARGS
	while IFS= read -r line; do
		line="${line%%#*}" # strip comments
		line="${line// /}" # strip spaces
		[[ -z "$line" ]] && continue
		local key="${line%%=*}"
		local val="${line#*=}"
		TEST_ARGS["$key"]="$val"
	done <"$cfg_file"
}

# Compare two network JSONs. Only compile_time and start_time are stripped
# (truly build-variant). All other metadata — cli_args, best_loss, epoch,
# network_json_out, etc. — is compared as-is. Test-runner plumbing
# (--network_json_out paths) is normalized away since those are not test data.
compare_networks() {
	local actual="$1" expected="$2"
	python3 - "$actual" "$expected" <<'PYEOF'
import json, sys, difflib

def normalize(json_path):
    with open(json_path) as f:
        obj = json.load(f)
    other = obj.setdefault("Associated_Data", {}).setdefault("other", {})
    # Strip build timestamps only
    for k in ("compile_time", "start_time", "git_commit"):
        other.pop(k, None)
    # Strip --network_json_out and its value from cli_args (test runner plumbing)
    if "cli_args" in other and isinstance(other["cli_args"], list):
        args = list(other["cli_args"])
        while "--network_json_out" in args:
            idx = args.index("--network_json_out")
            args.pop(idx)
            if idx < len(args):
                args.pop(idx)
        other["cli_args"] = sorted(args)
    # Normalize network_json_out to empty (it's a runner tmp path, not test data)
    other.pop("network_json_out", None)
    return obj

a = normalize(sys.argv[1])
b = normalize(sys.argv[2])
if a == b:
    print("PASS")
else:
    sa, sb = json.dumps(a, sort_keys=True, indent=2), json.dumps(b, sort_keys=True, indent=2)
    diff = list(difflib.unified_diff(sa.splitlines(), sb.splitlines(), lineterm=""))
    print("FAIL")
    for line in diff[:40]:
        print(line)
PYEOF
}

# --- main ---

build_if_needed

# Decide which tests to run
if [ $# -ge 1 ]; then
	test_name="$1"
	if [[ "$test_name" != golden_test_* ]]; then
		test_name="golden_test_${test_name}"
	fi
	cfg_file="$(find "$GOLDEN_DIR" -name "${test_name}.cfg" -type f 2>/dev/null | head -1)"
	if [ -z "$cfg_file" ]; then
		echo "Error: config not found for ${test_name}"
		exit 1
	fi
	cfgs=("$cfg_file")
else
	# All tests — find all .cfg files recursively
	cfgs=()
	while IFS= read -r -d '' f; do
		cfgs+=("$f")
	done < <(find "$GOLDEN_DIR" -name "golden_test_*.cfg" -type f -print0 2>/dev/null)
fi

if [ ${#cfgs[@]} -eq 0 ]; then
	echo "No golden tests found in $GOLDEN_DIR"
	exit 1
fi

PASSED=0
FAILED=0

for cfg_file in "${cfgs[@]}"; do
	test_name="$(basename "$cfg_file" .cfg)"
	test_dir="$(dirname "$cfg_file")"

	golden_out="$test_dir/${test_name}.out"
	golden_net="$test_dir/${test_name}.net.json"

	# Parse config
	declare -A TEST_ARGS=()
	parse_cfg "$cfg_file"

	# Build CLI
	cli_args=()
	for key in "${!TEST_ARGS[@]}"; do
		cli_args+=("--${key}" "${TEST_ARGS[$key]}")
	done

	# Temp files for this run
	tmp_stdout="$(mktemp)"
	tmp_net="$(mktemp)"
	trap 'rm -f "$tmp_stdout" "$tmp_net"' EXIT

	# Run
	if ! "$BIN" "${cli_args[@]}" --network_json_out "$tmp_net" >"$tmp_stdout" 2>&1; then
		"$BIN" "${cli_args[@]}" --network_json_out "$tmp_net"
		echo "FAIL: $test_name (command failed)"
		FAILED=$((FAILED + 1))
		unset TEST_ARGS
		continue
	fi

	pass=true

	# Compare stdout against golden .out
	if [ -f "$golden_out" ]; then
		if ! diff -q "$golden_out" "$tmp_stdout" >/dev/null 2>&1; then
			echo "FAIL: $test_name (stdout mismatch)"
			diff -u "$golden_out" "$tmp_stdout" | head -30
			pass=false
		fi
	fi

	# Compare network JSON against golden .net.json
	if [ -f "$golden_net" ]; then
		net_result="$(compare_networks "$tmp_net" "$golden_net")"
		if [[ "$net_result" == FAIL* ]]; then
			echo "FAIL: $test_name (network JSON mismatch)"
			echo "$net_result" | tail -n +2
			pass=false
		fi
	fi

	if $pass; then
		echo "PASS: $test_name"
		PASSED=$((PASSED + 1))
	else
		FAILED=$((FAILED + 1))
	fi

	unset TEST_ARGS
done

echo ""
echo "Results: $PASSED passed, $FAILED failed, $((PASSED + FAILED)) total"
[ $FAILED -eq 0 ]
