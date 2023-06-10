using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(Graph))]
public class GraphEditor : Editor
{
    public override void OnInspectorGUI()
    {
        Graph graph = (Graph)target;
        if (DrawDefaultInspector())
        {
            graph.Generate();
        }

        if (GUILayout.Button("Generate"))
        {
            graph.Generate();
        }

        if (GUILayout.Button("Clear"))
        {
            graph.Clear();
        }
    }
}
