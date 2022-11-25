// Copyright (C) Chelsea Jaggi, 2022.
//
// Update model with:
// python3 frugally-deep/keras_export/convert_model.py keras_model.h5 fdeep_model.json
// Compile with:
// g++ -I /usr/include/eigen3/ -I frugally-deep/include deep-tracker-driver.cpp -O3 -I FunctionalPlus/include -I json/include

#include <fdeep/fdeep.hpp>
int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(4)),
        std::vector<float>{1, 2, 3, 4})});
    std::cout << fdeep::show_tensors(result) << std::endl;
}
