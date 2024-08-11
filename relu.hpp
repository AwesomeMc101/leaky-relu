class ReLU_Activation {
public:
    std::vector<std::vector<double>> m_output{};
    std::vector<std::vector<double>> input{}, dinputs{};

     void leak_proto_forward(std::vector<std::vector<double>> inp);
    void leak_proto_backward(std::vector<std::vector<double>> dvals);
}
