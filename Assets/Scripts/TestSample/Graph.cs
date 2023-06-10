using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Graph : MonoBehaviour
{
    [SerializeField]
    Transform pointPrefab;
    List<Transform> cubes;

    private void CreateCube(NDArray pos)
    {
        var temp = pos.ToArray<double>();

        cubes.Add(Instantiate(pointPrefab));
        cubes[cubes.Count - 1].localPosition = new Vector3((float)temp[0], (float)temp[1], (float)temp[2]);
    }

    public void Clear()
    {
        if (cubes.Count > 0)
        {
            foreach (var cube in cubes)
            {
                DestroyImmediate(cube.gameObject);
            }
        }
        cubes.Clear();
    }

    public void Generate()
    {
        if (cubes.Count > 0)
        {
            foreach (var cube in cubes)
            {
                DestroyImmediate(cube.gameObject);
            }
        }
        cubes.Clear();

        if (pointPrefab != null)
        {
            var x = np.linspace(-2, 2, 21);
            var y = np.power(x - 1, 4) + 5 * np.power(x, 3) - 8 * x * x + 3 * x;
            var z = np.zeros(x.shape);
            var p = np.vstack(x, y, z).T;

            for (int i = 0; i < x.shape[0];  i++)
            {
                CreateCube(p[i]);
            }
        }
    }
}
