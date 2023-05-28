using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class MapGenerator : MonoBehaviour
{
    public enum DrawMode { NoiseMap, ColorMap };
    public DrawMode drawMode;

    public int mapWidth;
    public int mapHeight;
    public int seed;
    // add description to the inspector
    [Tooltip("The lower the value, the more zoomed in the map will be")]
    public float scale;
    // set minimum value for octaves to 1
    [Range(1, 10)]
    public int octaves = 4;
    public float persistance = .5f;
    public float lacunarity = 2.0f;
    public Vector2 offset = Vector2.zero;

    public TerrainType[] regions;


    public bool autoUpdate;

    public void GenerateMap()
    {
        float[,] noiseMap = Noise.GenerateNoiseMap(mapWidth, mapHeight, seed, scale, octaves, persistance, lacunarity, offset);

        Color[] colorMap = new Color[mapWidth * mapHeight];

        for (int y = 0; y < mapHeight; y++)
        {
            for (int x = 0; x < mapWidth; x++)
            {
                for(int z = 0; z < regions.Length; z++)
                {
                    regions[z].color.a = 1;
                    if (noiseMap[x, y] <= regions[z].height)
                    {
                        colorMap[y * mapWidth + x] = regions[z].color;
                        break;
                    }
                }
            }
        }

        MapDisplay display = FindObjectOfType<MapDisplay>();

        // switch draw mode
        switch (drawMode)
        {
            case DrawMode.NoiseMap:
                display.DrawNoiseMap(noiseMap);
                break;
            case DrawMode.ColorMap:
                display.DrawColorMap(colorMap, mapWidth, mapHeight);
                break;
            default:
                break;
        }


    }
    void Start()
    {
        GenerateMap();
    }

    private void OnValidate()
    {
        if (mapWidth < 1)
            mapWidth = 1;
        if (mapHeight < 1)
            mapHeight = 1;
        if (octaves < 1)
            octaves = 1;
    }
}

[System.Serializable]
public struct TerrainType
{
    public string name;
    public float height;
    public Color color;
}
