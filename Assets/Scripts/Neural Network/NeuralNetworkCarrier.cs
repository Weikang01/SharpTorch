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
        predictions = np.argmax(forward_output, 1);
        accuracy = np.mean((accuracy == predictions).astype(typeof(double)));

        Debug.Log("accuracy: " + accuracy.ToString());
    }

    public void Loss()
    {
        //loss_output = network.CalculateLoss(test_y, forward_output, loss);
        Debug.Log("loss: " + network.CalculateLoss(test_y, forward_output, loss));
    }
}

[System.Serializable]
public struct LayerStruct
{
    public Vector2Int inputShape;
    public Vector2Int outputShape;
    public ActivationFunction activationFunction;
    public DenseLayer.Initializer initializer;
}
