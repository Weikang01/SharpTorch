using NumSharp;
using System;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;


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

public abstract class Layer
{
    public NDArray inputs;
    public NDArray outputs;
    public NDArray inputs_error;
    public NDArray inputShape;
    public NDArray outputShape;

    public abstract NDArray Forward(NDArray x);

    public abstract NDArray Backward(NDArray x);

    public abstract void UpdateParams(Optimizer optimizer);

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
    public override void UpdateParams(Optimizer optimizer)
    {
        throw new NotImplementedException();
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

    public override void UpdateParams(Optimizer optimizer)
    {
        throw new NotImplementedException();
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

    public override void UpdateParams(Optimizer optimizer)
    {
        throw new NotImplementedException();
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
        inputs_error = (np.maximum(np.zeros(x.shape), x) == 0).astype(np.float32);
        return inputs_error;
    }

    public override string ErrorsToString()
    {
        return base.ErrorsToString();
    }

    public override void UpdateParams(Optimizer optimizer)
    {
        throw new NotImplementedException();
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

    public override void UpdateParams(Optimizer optimizer)
    {
        throw new NotImplementedException();
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

    public override void UpdateParams(Optimizer optimizer)
    {
        throw new NotImplementedException();
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
    public void Update(Layer layer)
    {

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
        //y = np.eye(y.shape[0], M: y[0]);
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

public class DenseLayer : Layer
{

    public NDArray weights_error;
    public NDArray biases_error;
    public NDArray weights;
    public NDArray biases;

    #region initializers
    public enum Initializer
    {
        Identity,
        Random,
        He,
        Xavier
    }
    
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

    public override void UpdateParams(Optimizer optimizer)
    {
        throw new NotImplementedException();
    }
}

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

    // backpropagation
    public void Backward(NDArray y, NDArray yHat)
    {
        y = loss.Backward(y, yHat);
        // backpropagate error
        for (int j = layers.Count - 1; j >= 0; j--)
            y = layers[j].Backward(y);
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