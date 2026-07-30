#include "network_utils.h"
#include "framework.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace std;
using nlohmann::json;

void load_network(neuro::Processor** pp, neuro::Network* net) {
    json proc_params;
    std::string proc_name;
    neuro::Processor* p;

    p = *pp;
    if (p == nullptr) {
        proc_params = net->get_data("proc_params");
        proc_name   = net->get_data("other")["proc_name"];
        p           = neuro::Processor::make(proc_name, proc_params);
        *pp         = p;
    }

    if (p->get_network_properties().as_json() !=
        net->get_properties().as_json()) {
        fprintf(stderr,
                "%s: load_network: Network and processor properties do not "
                "match.\n",
                __FILE__);
        exit(1);
    }

    if (!p->load_network(net)) {
        fprintf(stderr, "%s: load_network: Failed to load network.\n",
                __FILE__);
        exit(1);
    }
}

void export_network(neuro::Network* n, const CliConfig& cfg,
                    double best_train_acc, double best_train_loss,
                    double best_test_acc, double best_test_loss,
                    const std::vector<std::string>& label_strings,
                    bool is_regression) {
    if (cfg.network_json_out.empty()) {
        return;
    }

    nlohmann::json meta     = n->get_data("other");
    meta["best_train_loss"] = best_train_loss;
    meta["best_test_loss"]  = best_test_loss;
    meta["best_train_acc"]  = best_train_acc;
    meta["best_test_acc"]   = best_test_acc;
    meta["epoch"]           = cfg.epochs;
    meta["regression"]      = is_regression;

    // Export label mapping (classification only)
    if (!label_strings.empty()) {
        nlohmann::json labels_json;
        for (const auto& s : label_strings) {
            labels_json.push_back(s);
        }
        meta["label_mapping"] = labels_json;
    }

    n->set_data("other", meta);

    nlohmann::json j;
    n->to_json(j);
    std::ofstream fout(cfg.network_json_out);
    if (!fout) {
        fprintf(stderr, "Failed to open %s for writing\n",
                cfg.network_json_out.c_str());
        exit(1);
    }
    fout << j.dump(2) << std::endl;
    fout.close();
}
