using NumSharp;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetworkCarrier : MonoBehaviour
{
    public NDArray test_values;
    public LayerStruct[] test_layers;
    public SequentialNetwork network;
    public NDArray test_y;
    // generate a read only slot for the output
    public NDArray forward_output;
    public LossFunction loss;
    private NDArray predictions;
    private NDArray accuracy;
    public OptimizerStruct optimizerStruct;
    public int epochs;

    public void GenerateNeuralNetwork()
    {
        network = new SequentialNetwork();
        for (int i = 0;i < test_layers.Length;i++)
        {
            network.Add(new DenseLayer(
                new int[] { test_layers[i].inputShape.x, test_layers[i].inputShape.y },
                new int[] { test_layers[i].outputShape.x, test_layers[i].outputShape.y }, 
                test_layers[i].initializer));
            network.Add(Layer.GetActivationFunction(test_layers[i].activationFunction));
        }
        
        Debug.Log("Network Generated!");
    }

    public void Forward()
    {
        forward_output = network.Forward(test_values);

        Debug.Log("forward_output: " + forward_output.ToString());

        predictions = np.argmax(forward_output, 1);
        accuracy = np.mean((test_y == predictions).astype(typeof(double)));

        Debug.Log("accuracy: " + accuracy.ToString());
    }

    public void Loss()
    {
        Debug.Log("loss: " + network.CalculateLoss(test_y, forward_output, loss));
    }

    public void Optimize()
    {
        network.Optimize(optimizerStruct.optimizer, optimizerStruct.learningRate);
    }

    public void Train()
    {
        network.Train(test_values, test_y, epochs, optimizerStruct.learningRate, loss);
    }
}

[System.Serializable]
public struct LayerStruct
{
    public Vector2Int inputShape;
    public Vector2Int outputShape;
    public ActivationFunction activationFunction;
    public Initializer initializer;
}

[System.Serializable]
public struct OptimizerStruct
{
    public OptimizerFunction optimizer;
    public float learningRate;
    public float momentum;
    public float decay;
    public float nesterov;
}
