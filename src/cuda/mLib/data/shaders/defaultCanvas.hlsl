
cbuffer ConstantBufferCircle : register(b0)
{
	float4 color;
	float2 center;
	float radius;
	float dummy;
}

Texture2D modelTexture : register(t0);
SamplerState modelSampler : register(s0);

struct VertexShaderOutput
{
	float4 position : SV_POSITION;
	float2 texCoord : TEXCOORD0;
	float3 normal : NORMAL;
	float3 worldPos : WORLDPOS;
};

VertexShaderOutput circleVS(float4 position : position, float3 normal : normal,	float4 color : color, float2 texCoord : texCoord)
{
	VertexShaderOutput output;
	output.position = position;
	output.texCoord = texCoord.xy;
	output.normal = normal;
	output.worldPos = position.xyz;
	return output;
}

float4 circlePS(VertexShaderOutput input) : SV_Target
{
	float dist = length(input.position.xy - center.xy);
	if (dist > radius) {
		discard;
	}
	return float4(color.xyz, 1.0f);
}


VertexShaderOutput billboardVS(float4 position : position, float3 normal : normal, float4 color : color, float2 texCoord : texCoord)
{
	VertexShaderOutput output;
	output.position = position;
	output.texCoord = texCoord.xy;
	output.normal = normal;
	output.worldPos = position.xyz;
	return output;
}

float4 billboardPS(VertexShaderOutput input) : SV_Target
{
	float4 texColor = modelTexture.Sample(modelSampler, float2(input.texCoord.x, 1.0f - input.texCoord.y));
	if (texColor.w == 0.0f) {
		discard;
	}
	return texColor;
}

VertexShaderOutput boxVS(float4 position : position, float3 normal : normal, float4 color : color, float2 texCoord : texCoord)
{
	VertexShaderOutput output;
	output.position = position;
	//output.texCoord = texCoord.xy;
	//output.normal = normal;
	output.normal = color.xyz;
	output.texCoord = float2(color.w, 1.0f);
	output.worldPos = position.xyz;
	return output;
}

float4 boxPS(VertexShaderOutput input) : SV_Target
{
	return float4(input.normal.xyz, input.texCoord.x);
}
