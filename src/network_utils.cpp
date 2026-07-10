#include "network_utils.h"
#include "framework.hpp"
#include <cstdio>
#include <cstdlib>
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
