Shader "Unlit/CubeDemoShader"
{
    SubShader
    {
        CGPROGRAM
        #pragma surface ConfigureSurface Standard fullforwardshadows
        #pragma target 3.0

        struct Input
        {
			float3 worldPos;
		};

        void ConfigureSurface(Input input, inout SurfaceOutputStandard surface) 
        {
            surface.Smoothness = 0.5;
        }

        ENDCG
    }

    Fallback "Diffuse"
}
