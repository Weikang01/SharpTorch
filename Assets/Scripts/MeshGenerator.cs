using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class MeshGenerator
{
    public static void GenerateTerrainMesh(float[,] heightMap)
    {
        int width = heightMap.GetLength(0);
        int height = heightMap.GetLength(1);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {

            }
        }
    }
}

public class MeshData
{
    public Vector3[] vertices;
    public int[] indices;


    private int triangleIndex = 0; // index of the triangle we are currently working on

    public MeshData(int meshWidth, int meshHeight)
    {
        vertices = new Vector3[meshWidth * meshHeight];
        indices = new int[(meshWidth - 1) * (meshHeight - 1) * 6];
    }

    public void AddTriangle(int a, int b, int c)
    {
        indices[triangleIndex++] = a;
        indices[triangleIndex++] = b;
        indices[triangleIndex++] = c;
    }
}
