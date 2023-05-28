using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MapDisplay : MonoBehaviour
{
    public SpriteRenderer spriteRenderer;

    void ApplyTexture(Texture2D texture, int width, int height)
    {
        // get original size of sprite
        Rect rect = spriteRenderer.sprite.rect;

        // get scale of sprite
        Vector3 scale = spriteRenderer.transform.localScale;

        // apply texture to sprite
        spriteRenderer.sprite = Sprite.Create(texture, new Rect(0, 0, width, height), Vector2.one * .5f);

        spriteRenderer.transform.localScale = new Vector3(scale.x * rect.width / spriteRenderer.sprite.rect.width, scale.y * rect.height / spriteRenderer.sprite.rect.height, 1);
    }

    public void DrawNoiseMap(float[,] noiseMap)
    {
        int width = noiseMap.GetLength(0);
        int height = noiseMap.GetLength(1);

        Texture2D texture = new Texture2D(width, height);
        texture.filterMode = FilterMode.Point;
        texture.wrapMode = TextureWrapMode.Clamp;

        Color[] colors = new Color[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
                colors[y * width + x] = Color.Lerp(Color.black, Color.white, noiseMap[x, y]);
        }

        texture.SetPixels(colors);
        texture.Apply();

        ApplyTexture(texture, width, height);
    }

    public void DrawColorMap(Color[] colorMap, int width, int height)
    {
        // convert 1 bit color array of size height * width into sprite
        Texture2D texture = new Texture2D(width, height);
        texture.filterMode = FilterMode.Point;
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.SetPixels(colorMap);
        texture.Apply();

        ApplyTexture(texture, width, height);
    }
}
