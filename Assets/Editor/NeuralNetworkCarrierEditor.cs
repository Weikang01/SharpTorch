using UnityEngine;
using UnityEditor;


[CustomEditor(typeof(NeuralNetworkCarrier))]
public class NeuralNetworkCarrierEditor : Editor
{
    
    public override void OnInspectorGUI()
    {
        NeuralNetworkCarrier neuralNetworkCarrier = (NeuralNetworkCarrier)target;

        GUILayout.BeginVertical();

        DrawDefaultInspector();

        GUILayout.Space(2);

        //generate a button to generate the network
        if (GUILayout.Button("Generate Neural Network"))
        {
            neuralNetworkCarrier.test_values = new double[,] {
                { 1, -2, 3 },
                { 2, 0.5, -1 },
                { 3, 0, 0 },
                { 4, 1, 1 }
            };

            neuralNetworkCarrier.GenerateNeuralNetwork();

            ((DenseLayer)neuralNetworkCarrier.network.layers[0]).biases
                = new double[] { 1, 2, 3, 4 };
        }

        GUILayout.Space(2);

        if (GUILayout.Button("Show Layer Details"))
        {
            Debug.Log(neuralNetworkCarrier.network);
        }

        GUILayout.Space(2);

        // generate a button to forward the network
        if (GUILayout.Button("Forward"))
        {
            neuralNetworkCarrier.Forward();
        }

        GUILayout.Space(2);

        if (GUILayout.Button("Loss"))
        {
            neuralNetworkCarrier.test_y = new double[,]
            {
                { 0,1,0,0 },
                { 1,0,0,0 },
                { 0,0,1,0 },
                { 1,0,0,0 }
            };
            neuralNetworkCarrier.test_y = new double[] {1,0,2,0};

            neuralNetworkCarrier.Loss();
        }


        GUILayout.Space(2);

        // test backward propagation
        if (GUILayout.Button("Test Backward"))
            neuralNetworkCarrier.network.Backward(neuralNetworkCarrier.test_y, neuralNetworkCarrier.forward_output);


        GUILayout.Space(2);

        if (GUILayout.Button("Print Errors"))
        {
            Debug.Log(neuralNetworkCarrier.network.ErrorsToString());
        }

        GUILayout.EndVertical();
    }
}
