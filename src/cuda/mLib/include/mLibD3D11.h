#pragma once

//
// application-base headers
//
#include "application-base/windowWin32.h"
#include "application-base/graphicsAsset.h"
#include "application-base/graphicsDevice.h"
#include "application-base/applicationWin32.h"

//
// D3D11 headers
//
#pragma warning ( disable : 4005)

#include <d3d11.h>
#include <d3dx11.h>	//D3dX is in theory deprecated // see D3D11Utility.cpp, line 84
#include <d3dcompiler.h>

//
// application-d3d11 headers
//
#include "application-d3d11/D3D11VertexShader.h"
#include "application-d3d11/D3D11GeometryShader.h"
#include "application-d3d11/D3D11PixelShader.h"
#include "application-d3d11/D3D11ShaderManager.h"
#include "application-d3d11/D3D11GraphicsDevice.h"
#include "application-d3d11/D3D11Utility.h"
#include "application-d3d11/D3D11ConstantBuffer.h"
#include "application-d3d11/D3D11TriMesh.h"
#include "application-d3d11/D3D11PointCloud.h"
#include "application-d3d11/D3D11Texture2D.h"
#include "application-d3d11/D3D11Texture3D.h"
#include "application-d3d11/D3D11Buffer.h"
#include "application-d3d11/D3D11RenderTarget.h"
#include "application-d3d11/D3D11Canvas2D.h"
#include "application-d3d11/D3D11AssetRenderer.h"
