// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/convolution_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/network_helper.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginal(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    std::shared_ptr<ngraph::opset1::Constant> weights,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    const auto dequantization = makeDequantization(input, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((weights->cast_vector<float>().size() != 1ul) && (weights->cast_vector<float>().size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    if (weights->cast_vector<float>().size() == 1ul) {
        auto targetShape = ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 };
        weights = as_type_ptr<ngraph::opset1::Constant>(fold<ngraph::opset1::Broadcast>(
            weights, op::Constant::create(ngraph::element::i64, Shape{ targetShape.size() }, targetShape)));
    }

    const auto onWeights = fakeQuantizeOnWeights.empty() ? weights :
        ngraph::builder::makeFakeQuantize(
            weights, weights->get_element_type(),
            fakeQuantizeOnWeights.quantizationLevel,
            fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues,
            fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues,
            fakeQuantizeOnWeights.outputHighValues);

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ngraph::op::TemporaryReplaceOutputType(dequantization, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(onWeights, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});
    convolution->set_friendly_name("output");
    auto& rtInfo = convolution->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("convolution");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginalWithIncorrectWeights(
    const ngraph::Shape& inputShape,
    ngraph::element::Type precision,
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
    bool isCorrect) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fqOnData = fakeQuantizeOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
            fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];
    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    const auto fqOnWeights = fakeQuantizeOnWeights.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            weights, precision, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    const auto subtract = isCorrect ? nullptr : std::make_shared<DequantizationSubtract>(fqOnWeights,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, Shape{1, 1, 1, 1}, 3.0f));

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        fakeQuantizeOnData.empty() ? input : fqOnData,
        isCorrect ? fqOnWeights : subtract,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getReferenceWithIncorrectWeights(
    const ngraph::Shape& inputShape,
    ngraph::element::Type precision,
    ngraph::element::Type dataPrecision,
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
    ngraph::element::Type weightsPrecision,
    std::vector<float> weightsValues,
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter,
    bool isCorrect) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    std::shared_ptr<ngraph::opset1::FakeQuantize> fqOnData = as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        input,
        precision,
        fakeQuantizeOnData.quantizationLevel,
        fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues,
        fakeQuantizeOnData.inputHighValues,
        fakeQuantizeOnData.outputLowValues,
        fakeQuantizeOnData.outputHighValues));

    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fqOnData, dataPrecision);

    const auto deqBefore = dequantizationBefore.empty() ? nullptr : makeDequantization(fqOnData, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((weightsValues.size() != 1ul) && (weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    const std::shared_ptr<ngraph::Node> weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    const auto fqOnWeights = fakeQuantizeOnWeights.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            weights, precision, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    const auto subtract = isCorrect ? nullptr : std::make_shared<DequantizationSubtract>(fqOnWeights,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, 3.0f));

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ngraph::op::TemporaryReplaceOutputType(dequantizationBefore.empty() ? fqOnData : deqBefore, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(isCorrect ? weights : subtract, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});

    std::shared_ptr<ngraph::Node> multiply;
    if (!dequantizationAfter.multiply.empty()) {
        ngraph::Shape constShape = isCorrect ? Shape{ 1, 1, 1 } : Shape{ 1, 1, 1, 1 };
        multiply = std::make_shared<DequantizationMultiply>(convolution,
            std::make_shared<ngraph::opset1::Constant>(precision, constShape, dequantizationAfter.multiply.values[0]));
    }

    replace_node(fqOnData->get_input_node_shared_ptr(3),
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{}, fakeQuantizeOnData.outputLowValues[0]));

    replace_node(fqOnData->get_input_node_shared_ptr(4),
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{}, fakeQuantizeOnData.outputHighValues[0]));

    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fqOnData, dataPrecision);

    if (!dequantizationBefore.multiply.empty()) {
        ngraph::Shape constShape = isCorrect ? Shape{ 1, 1, 1 } : Shape{ 1, 1, 1, 1 };
        replace_node(
            deqBefore->get_input_node_shared_ptr(1),
            std::make_shared<ngraph::opset1::Constant>(precision, constShape, dequantizationBefore.multiply.values[0]));
    }

    if (isCorrect) {
        replace_node(
            weights,
            ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(weights, weightsPrecision));
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationAfter.empty() ? convolution : multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getReference(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    std::shared_ptr<ngraph::opset1::Constant> weights,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const ngraph::element::Type precisionAfterDequantization) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((weights->cast_vector<float>().size() != 1ul) && (weights->cast_vector<float>().size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    if (weights->cast_vector<float>().size() == 1ul) {
        auto targetShape = ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 };
        weights = as_type_ptr<ngraph::opset1::Constant>(fold<ngraph::opset1::Broadcast>(
            weights, op::Constant::create(ngraph::element::i64, Shape{ targetShape.size() }, targetShape)));
    }

    std::shared_ptr<ngraph::Node> onWeights = fakeQuantizeOnWeights.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(weights) :
        ngraph::builder::makeFakeQuantize(
            weights->output(0),
            weights->get_element_type(),
            fakeQuantizeOnWeights.quantizationLevel,
            fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues,
            fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues,
            fakeQuantizeOnWeights.outputHighValues);

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ngraph::op::TemporaryReplaceOutputType(deqBefore, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(onWeights, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});

    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(convolution, precisionAfterOperation);
    auto& rtInfo = convolution->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("convolution");

    const auto deqAfter = makeDequantization(convolution, dequantizationAfter);
    deqAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(deqAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::get(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precision,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& weightsValues,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const std::shared_ptr<ngraph::opset1::FakeQuantize> fqOnData = as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        input,
        precision,
        fakeQuantizeOnData.quantizationLevel,
        fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues,
        fakeQuantizeOnData.inputHighValues,
        fakeQuantizeOnData.outputLowValues,
        fakeQuantizeOnData.outputHighValues));

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];
    if ((weightsValues.size() != 1ul) && (weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    const std::shared_ptr<ngraph::Node> parentOnData = fakeQuantizeOnData.empty() ? std::dynamic_pointer_cast<ngraph::Node>(input) : fqOnData;

    const std::shared_ptr<ngraph::Node> weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    const std::shared_ptr<ngraph::Node> parentOnWeights = fakeQuantizeOnWeights.empty() ?
        weights :
        ngraph::builder::makeFakeQuantize(
            weights, precision, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ngraph::op::TemporaryReplaceOutputType(parentOnData, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(parentOnWeights, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    const std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
