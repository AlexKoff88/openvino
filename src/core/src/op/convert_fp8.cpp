// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/convert_fp8.hpp"

#include <memory>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/reference/convert.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::ConvertFP8);

op::ConvertFP8::ConvertFP8(const Output<Node>& arg)
    : Op({arg}) {
    constructor_validate_and_infer_types();
}

void op::ConvertFP8::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v0_Convert_validate_and_infer_types);
}

bool op::ConvertFP8::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v0_Convert_visit_attributes);
    return true;
}

shared_ptr<Node> op::ConvertFP8::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v0_Convert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ConvertFP8>(new_args.at(0));
}

namespace convertfp8 {
namespace {
template <element::Type_t INPUT_ET, element::Type_t OUTPUT_ET>
bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out)

{
    out->set_shape(arg->get_shape());
    size_t element_count = shape_size(out->get_shape());

    if (((INPUT_ET == element::u1) || (OUTPUT_ET == element::u1)) ||
        ((INPUT_ET == element::u4) || (OUTPUT_ET == element::u4)) ||
        ((INPUT_ET == element::i4) || (OUTPUT_ET == element::i4))) {
        runtime::reference::detail::lp_convert(arg->get_data_ptr<INPUT_ET>(),
                                               out->get_data_ptr<OUTPUT_ET>(),
                                               element_count,
                                               INPUT_ET,
                                               OUTPUT_ET);

        //TODO: Add convertion code here
    }
    return true;
}

#define TYPE_OUT_CASE(a, ...)                                     \
    case element::Type_t::a: {                                    \
        NGRAPH_OP_SCOPE(OV_PP_CAT3(evaluate_covert_out, _, a));   \
        rc = evaluate<INPUT_ET, element::Type_t::a>(__VA_ARGS__); \
    } break

template <element::Type_t INPUT_ET>
bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out) {
    bool rc = true;

    switch (out->get_element_type()) {
        TYPE_OUT_CASE(bf16, arg, out);
        TYPE_OUT_CASE(f16, arg, out);
        TYPE_OUT_CASE(f32, arg, out);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_convert(const HostTensorPtr& arg, const HostTensorPtr& out) {
    bool rc = true;
    switch (arg->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_convert, bf16, arg, out);
        NGRAPH_TYPE_CASE(evaluate_convert, f16, arg, out);
        NGRAPH_TYPE_CASE(evaluate_convert, f32, arg, out);
    default:
        rc = false;
        break;
    }
    return rc;
}


}  // namespace
}  // namespace convertfp8
bool op::v0::ConvertFP8::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    NGRAPH_OP_SCOPE(v0_Convert_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 1));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));
    return convertfp8::evaluate_convert(input_values[0], output_values[0]);
}

bool op::v0::ConvertFP8::has_evaluate() const {
    NGRAPH_OP_SCOPE(v0_Convert_has_evaluate);

    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
        break;
    default:
        return false;
    }
    switch (get_output_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
        break;
    default:
        return false;
    }
    return true;
}