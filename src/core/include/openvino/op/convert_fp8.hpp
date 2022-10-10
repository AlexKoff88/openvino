// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise type conversion operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ConvertFP8 : public Op {
public:
    OPENVINO_OP("ConvertFP8", "opset1");
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs a conversion operation.
    ConvertFP8() = default;
    /// \brief Constructs a conversion operation.
    ///
    /// \param arg          Node that produces the input tensor.
    /// \param destination_type  Element type for the output tensor.
    ConvertFP8(const Output<Node>& arg);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;


    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

protected:
    ov::element::Type m_destination_type;
};
}  // namespace v0
}  // namespace op
}  // namespace ov