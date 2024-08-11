
const long double leak_epsilon = 1e-3; //adjustable hyperparam

void ReLU_Activation::leak_proto_forward(std::vector<std::vector<double>> inp) {
    /* setting them to the leak epsilon instead of 0 prevents "dead neurons" */
    input = inp;
    m_output.clear();
  
    for (std::vector<double>& inp_v : inp) { //only pass as ref since we dont use row again + we arent calling reference in function params
        std::for_each(inp_v.begin(), inp_v.end(), [](double& val) {
            val = (val >= 0) ? val : (leak_epsilon*val); //if negative multiply by tiny minimal value so its nearly 0 but not 0
            });
        m_output.emplace_back(inp_v);
    }
}

void ReLU_Activation::leak_proto_backward(std::vector<std::vector<double>> dvals) {
    dinputs = dvals;
    for (int i = 0; i < input.size(); i++) { //loop through rows in inputs
        for (int x = 0; x < input[i].size(); x++) { //loop through the cols in each row
            if (input[i][x] < 0) {
                dinputs[i][x] = leak_epsilon; //as opposed to 0
            }
        }
    }
}
