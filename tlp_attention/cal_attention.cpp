#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;

void calAttention(py::array_t<double> attentions, py::array_t<double> inputs2KindIndex, py::array_t<double> results, py::array_t<double> results_cnt) {
    auto attentions_ptr = static_cast<double *>(attentions.request().ptr);
    auto inputs2KindIndex_ptr = static_cast<double *>(inputs2KindIndex.request().ptr);
    auto results_ptr = static_cast<double *>(results.request().ptr);
    auto results_cnt_ptr = static_cast<double *>(results_cnt.request().ptr);

    auto attentions_info = attentions.request();
    auto inputs2KindIndex_info = inputs2KindIndex.request();
    auto results_info = results.request();
    auto results_cnt_info = results_cnt.request();

    int batch_size = attentions_info.shape[0];
    int attentions_size1 = attentions_info.shape[1];
    int attentions_size2 = attentions_info.shape[2];

    std::cout << batch_size << attentions_size1 << attentions_size2 << std::endl;

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int i = 0; i < attentions_size1; ++i) {
            for (int j = 0; j < attentions_size2; ++j) {
                // std::cout << "batch:\t" << batch << "i:\t" << i << "j:\t" << j << std::endl;
                int ii = inputs2KindIndex_ptr[batch * attentions_size1 + i];
                int jj = inputs2KindIndex_ptr[batch * attentions_size2 + j];
                if(ii < 0 || jj < 0){
                    // std::cout << "-1" << "\t";
                    continue;
                }
                assert(ii < results_info.shape[0]);
                assert(jj < results_info.shape[1]);
                // std::cout << "ii:\t" << ii << "jj:\t" << jj << std::endl;
                results_ptr[ii * results_info.shape[1] + jj] += attentions_ptr[batch * attentions_size1 * attentions_size2 + i * attentions_size2 + j];
                results_cnt_ptr[ii * results_info.shape[1] + jj] += 1;
            }
        }
    }
}


PYBIND11_MODULE(cal_attention, m) {
    m.def("calAttention", &calAttention, "Function to perform the specified calculation");
}