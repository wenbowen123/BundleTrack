
cbuffer ConstantBuffer : register( b0 )
{
	matrix worldViewProj;
	matrix world;
	float4 lightDir;
	float4 eye;
}


struct VertexShaderOutput
{
    float4 position : SV_POSITION;
	float4 color : TEXCOORD0;
	float3 normal : TEXCOORD1;
};

VertexShaderOutput vertexShaderMain( float4 position : position,
									 float3 normal : normal,
									 float4 color : color,
									 float2 texCoord : texCoord)
{
    VertexShaderOutput output;
    output.position = mul( position, worldViewProj );
	output.color = color;
	output.normal = normal;
    return output;
}

float4 pixelShaderMain( VertexShaderOutput input ) : SV_Target
{
	//float3 n = normalize(input.normal) * 0.5 + float3(0.5, 0.5, 0.5);
    //return float4( n.x, n.y, n.z, 1.0f );
	return input.color;
}