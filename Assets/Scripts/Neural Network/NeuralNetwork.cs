using NumSharp;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.Security.Principal;
using Unity.VisualScripting;
using UnityEngine.Windows;
using System.Linq;

public abstract class Layer
{
    public NDArray inputs;
    public NDArray outputs;
    public NDArray inputs_error;

    public NDArray inputShape
    {
        get { return inputs.shape; }
        set { inputs = np.zeros(value.ToArray<int>()); }
    }

    public NDArray outputShape
    {
        get { return outputs.shape; }
        set { outputs = np.zeros(value.ToArray<int>()); }
    }

    public abstract NDArray Forward(NDArray x);

    public abstract NDArray Backward(NDArray x);

    public static Layer GetActivationFunction(ActivationFunction activationFunction)
    {
        switch (activationFunction)
        {
            case ActivationFunction.None:
                return new Activation_None();
            case ActivationFunction.Sigmoid:
                return new Activation_Sigmoid();
            case ActivationFunction.Tanh:
                return new Activation_Tanh();
            case ActivationFunction.ReLU:
                return new Activation_ReLU();
            case ActivationFunction.LeakyReLU:
                return new Activation_LeakyReLU();
            case ActivationFunction.Softmax:
                return new Activation_Softmax();
            default:
                return new Activation_None();
        }
    }

    //tostring
    public override string ToString()
    {
        return $"Layer: {GetType().Name}\n" +
            $"Input Shape: {inputShape}\n" +
            $"Output Shape: {outputShape}\n";
    }

    public virtual string ErrorsToString()
    {
        string str = "\nLayer: " + GetType().Name + "\n";
        str += "inputs_error: " + inputs_error.ToString() + "\n";
        return str;
    }
}

#region activation functions
public enum ActivationFunction
{
    None,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Softmax
}

internal abstract class Activation_Layer : Layer {}

internal class Activation_None : Layer
{
    public override NDArray Forward(NDArray x)
    {
        inputs = x;
        outputs = x;
        return x;
    }
    public override NDArray Backward(NDArray x)
    {
        inputs_error = x;
        return x;
    }

    public override string ErrorsToString()
    {
        return base.ErrorsToString();
    }
}

internal class Activation_Sigmoid : Layer
{
    public override NDArray Forward(NDArray x)
    {
        inputs = x;
        outputs = 1 / (1 + np.exp(np.zeros(x.shape) - x));
        return outputs;
    }

    public override NDArray Backward(NDArray x)
    {
        inputs_error = Forward(x) * (1 - Forward(x));
        return inputs_error;
    }

    public override string ErrorsToString()
    {
        return base.ErrorsToString();
    }
}

internal class Activation_Tanh : Layer
{
    public override NDArray Forward(NDArray x)
    {
        inputs = x;
        outputs = (np.exp(x) - np.exp(np.zeros(x.shape) - x)) / (np.exp(x) + np.exp(np.zeros(x.shape) - x));
        return outputs;
    }

    public override NDArray Backward(NDArray x)
    {
        inputs_error = 1 - np.square(x);
        return inputs_error;
    }

    public override string ErrorsToString()
    {
        return base.ErrorsToString();
    }
}

internal class Activation_ReLU : Layer
{
    public override NDArray Forward(NDArray x)
    {   
        inputs = x;
        outputs = np.maximum(np.zeros(x.shape), x);
        return outputs;
    }
    public override NDArray Backward(NDArray x)
    {
        //Debug.Log(x.ToString());

        inputs_error = np.maximum(np.zeros(x.shape), x);

        //Debug.Log((np.maximum(np.zeros(x.shape), x)).ToString());

        return inputs_error;
    }

    public override string ErrorsToString()
    {
        return base.ErrorsToString();
    }
}

internal class Activation_LeakyReLU : Layer
{
    public override NDArray Forward(NDArray x)
    {
        inputs = x;
        outputs = np.maximum(0.01f * x, x);
        return outputs;
    }
    public override NDArray Backward(NDArray x)
    {
        NDArray positive = (np.maximum(np.zeros(x.shape), x) == 0).astype(np.float32);
        inputs_error = positive + (np.ones(x.shape) - positive) * .01f;
        return inputs_error;
    }

    public override string ErrorsToString()
    {
        return base.ErrorsToString();
    }
}

internal class Activation_Softmax : Layer
{
    public override NDArray Forward(NDArray x)
    {
        inputs = x;

        NDArray x_max = np.max(x, axis: 1, keepdims: true);
        
        NDArray exp_values = np.exp(inputs - x_max);

        NDArray sum_exp = x.shape[1] * np.mean(exp_values, axis: 1, keepdims: true);

        outputs = exp_values / sum_exp;

        return outputs;
    }
    public override NDArray Backward(NDArray x)
    {
        List<NDArray> dinputs = new List<NDArray>();

        for (int i=0; i < outputs.shape[0]; i++)
        {
            NDArray cur_row = outputs[i];
            NDArray cur_dval = x[i];


            if (cur_row.ndim == 1)
                cur_row = cur_row.reshape(new int[] { 1, cur_row.shape[0] });
            if (cur_dval.ndim == 1)
                cur_dval = cur_dval.reshape(new int[] { 1, cur_dval.shape[0] });

            NDArray jacobian_matrix = cur_row * np.eye(cur_row.shape[1]) - np.dot(cur_row.T, cur_row);

            dinputs.Add(np.dot(cur_dval, jacobian_matrix)[0]);
        }

        inputs_error = np.vstack(dinputs.ToArray());
        
        return inputs_error;
    }

    public override string ErrorsToString()
    {
        return base.ErrorsToString();
    }
}
#endregion

#region optimizers
// optimizer enum
public enum OptimizerFunction
{
    None,
    SGD,
    Momentum,
    NAG,
    Adagrad,
    RMSprop,
    Adam
}

public abstract class Optimizer
{
    public static Optimizer GetOptimizer(OptimizerFunction optimizerFunction, float learning_rate)
    {
        switch (optimizerFunction)
        {
            case OptimizerFunction.None:
                return new Optimizer_None();
            case OptimizerFunction.SGD:
                return new Optimizer_SGD(learning_rate);
            default:
                return new Optimizer_None();
        }
    }

    public void Update(Layer layer)
    {
        switch (layer)
        {
            case DenseLayer dense:
                Update(dense);
                break;
            default:
                break;
        }
    }

    public abstract void Update(DenseLayer layer);
}

internal class Optimizer_None : Optimizer
{
    public override void Update(DenseLayer layer)
    {
        return;
    }
}

internal class Optimizer_SGD : Optimizer
{
    public float learning_rate;
    public float decay;
    public int iterations;
    public Optimizer_SGD(float learning_rate, float decay = 0.0f)
    {
        this.learning_rate = learning_rate;
        this.decay = decay;
        iterations = 0;
    }

    void UpdateLearningRate()
    {
        learning_rate = learning_rate * (1.0f / (1.0f + decay * iterations++));
    }

    public override void Update(DenseLayer layer)
    {
        if (this.decay > 0.0f)
            UpdateLearningRate();

        layer.weights -= learning_rate * layer.weights_error;
        layer.biases -= learning_rate * layer.biases_error;
    }
}

#endregion

#region loss functions
// loss function enum
public enum LossFunction
{
    None,
    MSE,
    MAE,
    CategoricalCrossEntropy
}

public abstract class Loss
{
    public NDArray inputs_error;

    public static Loss GetLossFunction(LossFunction lossFunction)
    {
        switch (lossFunction)
        {
            case LossFunction.None:
                return new Loss_None();
            case LossFunction.MSE:
                return new Loss_MSE();
            case LossFunction.MAE:
                return new Loss_MAE();
            case LossFunction.CategoricalCrossEntropy:
                return new Loss_CategoricalCrossEntropy();
            default:
                return new Loss_None();
        }
    }

    public abstract NDArray Forward(NDArray y, NDArray yHat);

    public abstract NDArray Backward(NDArray y, NDArray yHat);
}

internal class Loss_None : Loss
{
    public override NDArray Forward(NDArray y, NDArray yHat)
    {
        throw new NotImplementedException();
    }

    public override NDArray Backward(NDArray y, NDArray yHat)
    {
        throw new NotImplementedException();
    }
}

internal class Loss_MAE : Loss
{
    public override NDArray Forward(NDArray y, NDArray yHat)
    {
        throw new NotImplementedException();
    }

    public override NDArray Backward(NDArray y, NDArray yHat)
    {
        throw new NotImplementedException();
    }
}

internal class Loss_MSE : Loss
{
    public override NDArray Forward(NDArray y, NDArray yHat)
    {
        throw new NotImplementedException();
    }

    public override NDArray Backward(NDArray y, NDArray yHat)
    {
        throw new NotImplementedException();
    }
}

// implement optimizers
internal class Loss_CategoricalCrossEntropy : Loss
{
    public override NDArray Forward(NDArray y, NDArray yHat)
    {
        yHat = np.clip(yHat, 1e-7, 1 - 1e-7);
        NDArray confidences;

        if (y.ndim == 1)
        {
            List<NDArray> temp = new List<NDArray>();
            for (int i = 0; i < y.shape[0]; i++)
                temp.Add(yHat[i, y[i]]);

            confidences = np.concatenate(temp.ToArray());
        }
        else  // only for one-hot encoding
            confidences = yHat.shape[1] * np.mean(yHat * y, axis: 1);

        NDArray negative_log_likelihoods = np.mean(np.log(confidences));

        return np.zeros(negative_log_likelihoods.shape) - negative_log_likelihoods;
    }

    public override NDArray Backward(NDArray y, NDArray yHat)
    {
        if (y.ndim == 1)
        {
            List<NDArray> temp = new List<NDArray>();
            for (int i = 0; i < y.shape[0]; i++)
                temp.Add((np.arange(yHat.shape[1]) == y[i]).astype(np.int32));
            y = np.vstack(temp.ToArray());
        }

        inputs_error = y / yHat;
        inputs_error = (np.zeros(inputs_error.shape) - inputs_error) / yHat.shape[0];

        return inputs_error;
    }
}
#endregion

public enum Initializer
{
    Identity,
    Random,
    He,
    Xavier
}

#region dense layer
public class DenseLayer : Layer
{

    public NDArray weights_error;
    public NDArray biases_error;
    public NDArray weights;
    public NDArray biases;

    #region initializers
    
    static Func<DenseLayer, bool> GetInitializer(Initializer initializer)
    {
        switch (initializer)
        {
            case Initializer.Identity:
                return Identity;
            case Initializer.Random:
                return Random;
            case Initializer.He:
                return He;
            case Initializer.Xavier:
                return Xavier;
            default:
                return Random;
        }
    }

    static bool Identity(DenseLayer dense)
    {
        NDArray weight_shape = np.concatenate((dense.inputShape["1:"], dense.outputShape["1:"]));
        dense.weights = np.ones(weight_shape.ToArray<int>());
        dense.biases = np.zeros(dense.outputShape["1:"].ToArray<int>());
        return true;
    }

    static bool Random(DenseLayer dense)
    {
        NDArray weight_shape = np.concatenate((dense.inputShape["1:"], dense.outputShape["1:"]));
        dense.weights = np.random.uniform(-1, 1, weight_shape.ToArray<int>());
        dense.biases = np.random.uniform(-1, 1, dense.outputShape["1:"].ToArray<int>());
        return true;
    }

    static bool He(DenseLayer dense)
    {
        NDArray weight_shape = np.concatenate((dense.inputShape["1:"], dense.outputShape["1:"]));
        dense.weights = np.random.randn(weight_shape.ToArray<int>()) * np.sqrt(2 / dense.inputShape[0]);
        dense.biases = np.zeros(dense.outputShape["1:"].ToArray<int>());
        return true;
    }

    static bool Xavier(DenseLayer dense)
    {
        NDArray weight_shape = np.concatenate((dense.inputShape["1:"], dense.outputShape["1:"]));
        dense.weights = np.random.randn(weight_shape.ToArray<int>()) * np.sqrt(1 / dense.inputShape[0]);
        dense.biases = np.zeros(dense.outputShape["1:"].ToArray<int>());
        return true;
    }
    #endregion

    public DenseLayer(NDArray inputShape, NDArray outputShape, Initializer initializer = Initializer.Random)
    {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        GetInitializer(initializer)(this);
    }

    public override NDArray Forward(NDArray x)
    {
        inputs = x;
        outputs = np.dot(x, weights) + biases;

        return outputs;
    }

    public override NDArray Backward(NDArray x)
    {
        inputs_error = np.dot(x, weights.T);
        weights_error = np.dot(inputs.T, x);
        biases_error = biases.shape[0] * np.mean(x, axis:0);

        return inputs;
    }

    // tostring
    public override string ToString()
    {
        string str = "Dense Layer: " + inputShape.ToString() + " -> " + outputShape.ToString();
        str += "\nWeights:\n" + weights.ToString();
        str += "\nBiases: " + biases.ToString();

        return str;
    }

    public override string ErrorsToString()
    {
        string str = "\nLayer: " + GetType().Name + "\n";
        str += "inputs_error: " + inputs_error.ToString();
        str += "\nweights_error:\n" + weights_error.ToString();
        str += "\nbiases_error: " + biases_error.ToString();
        return str;
    }
}
#endregion

#region convolution layer
public class Conv2DLayer : Layer
{
    private NDArray kernel;
    private NDArray biases;
    public NDArray kernel_error;
    public NDArray biases_error;

    private int[] stride;
    private int[] padding;
    #region initializers
    static Func<Conv2DLayer, bool> GetInitializer(Initializer initializer)
    {
        switch (initializer)
        {
            case Initializer.Identity:
                return Identity;
            case Initializer.Random:
                return Random;
            default:
                return Random;
        }
    }

    static bool Identity(Conv2DLayer conv2DLayer)
    {
        conv2DLayer.kernel = np.ones(conv2DLayer.kernel.shape) / (np.prod(conv2DLayer.kernel.shape));
        return true;
    }

    static bool Random(Conv2DLayer conv2DLayer)
    {
        conv2DLayer.kernel = np.random.uniform(-1, 1, conv2DLayer.kernel.shape);
        conv2DLayer.biases = np.random.uniform(-1, 1, conv2DLayer.biases.shape);
        return true;
    }

    #endregion

    // input shape: (batch, channels, height, width)
    // output shape: (channels)
    // kernel shape: (height, width)

    // stride: (height, width)
    // padding: (height, width)
    public Conv2DLayer(NDArray inputShape, int outputChannels, Initializer initializer, NDArray kernelShape, NDArray stride, NDArray padding)
    {
        this.inputShape = inputShape;

        outputShape = ((inputShape["2:4"] - kernelShape + 2 * padding) / stride + 1).astype(np.int32);

        outputShape = np.concatenate(
            (inputShape[0].ToArray<int>(),
            np.array(outputChannels).reshape(1),
            outputShape
            ));

        outputs = np.zeros(outputShape.ToArray<int>());

        kernelShape = np.concatenate((new int[] { outputChannels }, inputShape[1].ToArray<int>(), kernelShape));

        kernel = np.ones(kernelShape.ToArray<int>());
        biases = np.zeros(outputChannels);

        this.stride = stride.ToArray<int>();
        this.padding = padding.ToArray<int>();
        GetInitializer(initializer)(this);
    }

    NDArray GeneratePatch(in NDArray cur_inputs, int batch, int input_y, int input_x)
    {
        NDArray samples = np.zeros(kernel["0,:,:,:"].shape);

        int[] v_params = { 0, 0, 0, 0 };
        int[] z_params = { 0, 0, 0, 0 };

        if (input_y >= 0)
            v_params[0] = input_y;
        else
            z_params[0] = -input_y;

        if (input_y + kernel.shape[2] >= inputShape[2])
        {
            v_params[1] = inputShape[2];
            z_params[1] = inputShape[2] - input_y;
        }
        else
        {
            v_params[1] = input_y + kernel.shape[2];
            z_params[1] = kernel.shape[2];
        }

        if (input_x >= 0)
            v_params[2] = input_x;
        else
            z_params[2] = -input_x;

        if (input_x + kernel.shape[3] >= inputShape[3])
        {
            v_params[3] = inputShape[3];
            z_params[3] = inputShape[3] - input_x;
        }
        else
        {
            v_params[3] = input_x + kernel.shape[3];
            z_params[3] = kernel.shape[3];
        }

        //Debug.Log(String.Format(":,{0}:{1},{2}:{3}\n{4},:,{5}:{6},{7}:{8}", z_params[0], z_params[1], z_params[2], z_params[3], batch, v_params[0], v_params[1], v_params[2], v_params[3]));


        samples[String.Format(":,{0}:{1},{2}:{3}", z_params[0], z_params[1], z_params[2], z_params[3])] = cur_inputs[String.Format("{0},:,{1}:{2},{3}:{4}", batch, v_params[0], v_params[1], v_params[2], v_params[3])];

        return samples;
    }

    public override NDArray Forward(NDArray x)
    {
        inputs = x;

        for (int batch = 0; batch < outputShape[0]; batch++)  // image_size
        {
            for (int out_channel = 0; out_channel < outputShape[1]; out_channel++)  // output_channel
            {
                for (int output_y = 0, input_y = -padding[0]; output_y < outputShape[2]; output_y ++, input_y+= stride[0])
                {
                    if (input_y + kernel.shape[2] < 0)
                        continue;

                    for (int output_x = 0, input_x = -padding[1]; output_x < outputShape[3]; output_x++, input_x += stride[1])
                    {
                        if (input_x + kernel.shape[3] < 0)
                            continue;

                        NDArray samples = GeneratePatch(x, batch, input_y, input_x) * kernel[String.Format("{0},:,:,:", out_channel)];

                        if (inputShape[1] == outputShape[1])
                        {
                            samples = np.mean(samples, axis: 1) * samples.shape[1];
                            samples = np.mean(samples, axis: 1) * samples.shape[1];

                            outputs[String.Format("{0},:,{1},{2}", batch, output_y, output_x)] = samples;
                        }
                        else
                        {
                            samples = np.mean(samples) * kernel.size;

                            // image_size, output_channel, height, width
                            outputs[String.Format("{0},{1},{2},{3}", batch, out_channel, output_y, output_x)] = samples + biases[out_channel];
                        }
                    }
                }
            }
        }

        return outputs;
    }

    public override NDArray Backward(NDArray x)
    {
        inputs_error = x;

        biases_error = x.mean(axis: 0) * x.shape[0];
        biases_error = biases_error.mean(axis: 1) * biases_error.shape[1];
        biases_error = biases_error.mean(axis: 1) * biases_error.shape[1];

        kernel_error = np.zeros(kernel.shape);
        biases_error = np.zeros(biases.shape);

        for (int batch = 0; batch < inputs.shape[0]; batch++)
        {
            for (int output_y = 0, input_y = -padding[0]; output_y < outputs.shape[2]; output_y++, input_y += stride[0])
            {
                if (input_y + stride[0] < 0)
                    continue;

                for (int output_x = 0, input_x = -padding[1]; output_x < outputs.shape[3]; output_x++, input_x += stride[1])
                {
                    if (input_x + stride[1] < 0)
                        continue;

                    for (int out_channel = 0; out_channel < kernel.shape[0]; out_channel++)
                    {
                        NDArray samples = GeneratePatch(inputs, batch, input_y, input_x);

                        NDArray cur_x = x[String.Format("{0},:,{1},{2}", batch, output_y, output_x)];

                        for (int in_channel = 0; in_channel < kernel.shape[1]; in_channel++)
                            samples[in_channel] *= cur_x[in_channel];

                        kernel_error[String.Format("{0},:,:,:", out_channel)] += samples;

                        biases_error[out_channel] += np.mean(samples) * samples.size;
                    }
                }
            }
        }

        return inputs_error;
    }
}
#endregion

#region pooling
public class MaxPooling : Layer
{
    private int[] window_shape;
    private int[] stride;
    private int[] padding;

    //public NDArray inputs;
    //public NDArray outputs;
    //public NDArray inputs_error;
    //public NDArray inputShape;
    //public NDArray outputShape;

    public MaxPooling(int[] window_shape, int[] stride, int[] padding)
    {
        this.window_shape = window_shape;
        this.padding = padding;
        this.stride = stride;
    }

    NDArray GeneratePatch(in NDArray cur_inputs, int input_y, int input_x)
    {
        NDArray s_shape = np.vstack(inputShape[":2"], new NDArray(window_shape));
        NDArray samples;
        samples = np.ones(s_shape.ravel().ToArray<int>());
        samples *= double.MinValue;


        int[] v_params = { 0, 0, 0, 0 };
        int[] z_params = { 0, 0, 0, 0 };

        if (input_y >= 0)
            v_params[0] = input_y;
        else
            z_params[0] = -input_y;

        if (input_y + window_shape[0] >= inputShape[2])
        {
            v_params[1] = inputShape[2];
            z_params[1] = inputShape[2] - input_y;
        }
        else
        {
            v_params[1] = input_y + window_shape[0];
            z_params[1] = window_shape[0];
        }

        if (input_x >= 0)
            v_params[2] = input_x;
        else
            z_params[2] = -input_x;

        if (input_x + window_shape[1] >= inputShape[3])
        {
            v_params[3] = inputShape[3];
            z_params[3] = inputShape[3] - input_x;
        }
        else
        {
            v_params[3] = input_x + window_shape[1];
            z_params[3] = window_shape[1];
        }

        //Debug.Log(String.Format(":,{0}:{1},{2}:{3}\n{4},:,{5}:{6},{7}:{8}", z_params[0], z_params[1], z_params[2], z_params[3], batch, v_params[0], v_params[1], v_params[2], v_params[3]));

        //Debug.Log(cur_inputs[String.Format("{0},:,{1}:{2},{3}:{4}", batch, v_params[0], v_params[1], v_params[2], v_params[3])].ToString());

        samples[String.Format(":,:,{0}:{1},{2}:{3}", z_params[0], z_params[1], z_params[2], z_params[3])] = cur_inputs[String.Format(":,:,{0}:{1},{2}:{3}", v_params[0], v_params[1], v_params[2], v_params[3])];

        return samples;
    }


    // input shape: (batch, channels, height, width)
    public override NDArray Forward(NDArray x)
    {
        inputs = x;

        outputShape = ((inputShape["2:4"] - (new NDArray(window_shape) - 1) + 2 * new NDArray(padding) - 1) / stride + 1).astype(np.int32);
        outputShape = np.concatenate(
            (inputShape[0].ToArray<int>(),
            inputShape[1].ToArray<int>(),
            outputShape
            ));

        for (int output_y = 0, input_y = -padding[0]; output_y < outputShape[2]; output_y++, input_y += stride[0])
        {
            if (input_y + window_shape[0] < 0)
                continue;

            for (int output_x = 0, input_x = -padding[1]; output_x < outputShape[3]; output_x++, input_x += stride[1])
            {
                if (input_x + window_shape[1] < 0)
                    continue;

                NDArray samples = GeneratePatch(x, input_y, input_x);

                samples = np.max(samples, axis: 2);
                samples = np.max(samples, axis: 2);

                outputs[String.Format(":,:,{0},{1}", output_y, output_x)] = samples;
            }
        }

        return outputs;
    }

    public override NDArray Backward(NDArray x)
    {
        inputs_error = np.zeros(inputs.shape);

        for (int output_y = 0, input_y = -padding[0]; output_y < x.shape[2]; output_y++, input_y += stride[0])
        {
            if (input_y + window_shape[0] < 0)
                continue;

            for (int output_x = 0, input_x = -padding[1]; output_x < x.shape[3]; output_x++, input_x += stride[1])
            {
                if (input_x + window_shape[1] < 0)
                    continue;

                int[] v_params = { 0, 0, 0, 0 };
                int[] z_params = { 0, 0, 0, 0 };

                if (input_y >= 0)
                    v_params[0] = input_y;
                else
                    z_params[0] = -input_y;

                if (input_y + window_shape[0] >= inputShape[2])
                {
                    v_params[1] = inputShape[2];
                    z_params[1] = inputShape[2] - input_y;
                }
                else
                {
                    v_params[1] = input_y + window_shape[0];
                    z_params[1] = window_shape[0];
                }

                if (input_x >= 0)
                    v_params[2] = input_x;
                else
                    z_params[2] = -input_x;

                if (input_x + window_shape[1] >= inputShape[3])
                {
                    v_params[3] = inputShape[3];
                    z_params[3] = inputShape[3] - input_x;
                }
                else
                {
                    v_params[3] = input_x + window_shape[1];
                    z_params[3] = window_shape[1];
                }

                for (int batch = 0; batch < x.shape[0]; batch++)
                {
                    for (int in_channel = 0; in_channel < x.shape[1]; in_channel++)
                    {
                        NDArray cur_patch = inputs[String.Format("{0},{1},{2}:{3},{4}:{5}", batch, in_channel, v_params[0], v_params[1], v_params[2], v_params[3])];

                        int max_idx = cur_patch.argmax();

                        inputs_error[String.Format("{0},{1},{2},{3}", batch, in_channel, v_params[0] + (max_idx / inputShape[2]), v_params[2] + (max_idx % inputShape[2]))] = x[String.Format("{0},{1},{2},{3}", batch, in_channel, output_y, output_x)];
                    }
                }
            }
        }

        return inputs_error;
    }
}
#endregion




//////////////////////
//////          //////
////// Network ///////
//////          //////
//////////////////////

public class SequentialNetwork
{
    Loss loss;

    // list of layers
    public List<Layer> layers = new List<Layer>();
    public SequentialNetwork() { }

    public void Add(Layer layer)
    {
        layers.Add(layer);
    }

    // fit n dimension of float 
    public NDArray Forward(NDArray x)
    {
        for (int j = 0; j < layers.Count; j++)
            x = layers[j].Forward(x);

        return x;
    }

    // calculate loss
    public NDArray CalculateLoss(NDArray y, NDArray yHat, LossFunction lossFunction)
    {
        loss = Loss.GetLossFunction(lossFunction);
        return loss.Forward(y, yHat);
    }

    public NDArray CalculateCategoricalAccuracy(NDArray y, NDArray yHat)
    {
        if (y.ndim == 2)
            y = np.argmax(y, 1);

        return np.mean((y == np.argmax(yHat, 1)).astype(typeof(double)));
    }

    // backpropagation
    public void Backward(NDArray y, NDArray yHat)
    {
        y = loss.Backward(y, yHat);
        // backpropagate error
        for (int j = layers.Count - 1; j >= 0; j--)
            y = layers[j].Backward(y);
    }

    public void Optimize(OptimizerFunction optimizerFunction, float learning_rate)
    {
        Optimizer optimizer = Optimizer.GetOptimizer(optimizerFunction, learning_rate);
        for (int j = 0; j < layers.Count; j++)
            optimizer.Update(layers[j]);
    }

    public void Train(NDArray x, NDArray y, int iter, float learning_rate, LossFunction lossFunction = LossFunction.None)
    {
        for (int i = 1; i <= iter; i++)
        {
            NDArray yHat = Forward(x);
            //Debug.Log("yHat: " + yHat.ToString());
            NDArray loss = CalculateLoss(y, yHat, lossFunction);
            NDArray acc = CalculateCategoricalAccuracy(y, yHat);
            Backward(y, yHat);
            Optimize(OptimizerFunction.SGD, learning_rate);

            //Debug.Log("y: " + y.ToString() + " yHat: " + yHat.ToString());

            Debug.Log("Iteration: " + i + " Loss: " + loss.ToString() + " Accuracy: " + acc.ToString());
        }
    }

    // tostring
    public override string ToString()
    {
        string str = "";
        for (int j = 0; j < layers.Count; j++)
            str += layers[j].ToString() + "\n";
        return str;
    }

    public string ErrorsToString()
    {
        string str = "";
        for (int j = 0; j < layers.Count; j++)
            str += layers[j].ErrorsToString() + "\n";
        return str;
    }
}