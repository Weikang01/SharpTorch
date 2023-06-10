using UnityEngine;
using UnityEditor;
using NumSharp;

[CustomEditor(typeof(NeuralNetworkCarrier))]
public class NeuralNetworkCarrierEditor : Editor
{
    public override void OnInspectorGUI()
    {
        NeuralNetworkCarrier neuralNetworkCarrier = (NeuralNetworkCarrier)target;
        neuralNetworkCarrier.test_values = new double[,] {
                { 1, -2, 3 },
                { 2, 0.5, -1 },
                { 3, 0, 0 },
                { 4, 1, 1 }
            };  // test x
        neuralNetworkCarrier.test_y = new double[] { 1, 0, 2, 0 };  // test y

        GUILayout.BeginVertical();

        DrawDefaultInspector();

        GUILayout.Space(2);

        if (GUILayout.Button("Test Convolution 2D"))
        {
            Conv2DLayer conv2DLayer = new Conv2DLayer(
                new int[] { 1, 3, 16, 16 },  // #image, #input_channel, #height, #width
                4,                           // #output_channel
                Initializer.Identity, 
                new int[] { 3, 3 }, // kernel
                new int[] { 1, 1 }, // stride
                new int[] { 1, 1 }  // padding
                );

            MaxPooling maxPooling = new MaxPooling(
                new int[] { 2, 2 }, // window size
                new int[] { 2, 2 }, // stride
                new int[] { 1, 1 }  // padding
                );

            // NDArray test_images = np.random.uniform(0, 1, new int[] { 2, 16, 16, 2 });
            NDArray test_images = np.ones(new int[] { 1, 3, 16, 16 });
            conv2DLayer.Forward(test_images);
            maxPooling.Forward(conv2DLayer.outputs);
            maxPooling.Backward(maxPooling.outputs);
            conv2DLayer.Backward(conv2DLayer.outputs);

            //Debug.Log(conv2DLayer.outputs.ToString());
        }

        GUILayout.Space(20);

        //generate a button to generate the network
        if (GUILayout.Button("Generate Neural Network"))
        {
            neuralNetworkCarrier.GenerateNeuralNetwork();
        }

        GUILayout.Space(2);

        if (GUILayout.Button("Train"))
        {
            neuralNetworkCarrier.Train();
        }

        GUILayout.Space(10);

        if (GUILayout.Button("Show Layer Details"))
        {
            Debug.Log(neuralNetworkCarrier.network);
        }

        GUILayout.Space(2);

        //generate a button to forward the network
        if (GUILayout.Button("Forward"))
        {
            neuralNetworkCarrier.Forward();
        }

        GUILayout.Space(2);

        if (GUILayout.Button("Calculate Loss"))
        {
            neuralNetworkCarrier.Loss();
        }

        GUILayout.Space(2);

        //test backward propagation
        if (GUILayout.Button("Backward"))
            neuralNetworkCarrier.network.Backward(neuralNetworkCarrier.test_y, neuralNetworkCarrier.forward_output);

        GUILayout.Space(2);

        if (GUILayout.Button("Print Errors"))
        {
            Debug.Log(neuralNetworkCarrier.network.ErrorsToString());
        }

        GUILayout.Space(2);

        if (GUILayout.Button("Optimize"))
        {
            neuralNetworkCarrier.Optimize();
        }


        GUILayout.EndVertical();
    }
}
