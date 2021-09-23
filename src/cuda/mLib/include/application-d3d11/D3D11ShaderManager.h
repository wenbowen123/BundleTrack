
#ifndef APPLICATION_D3D11_D3D11SHADERMANAGER_H_
#define APPLICATION_D3D11_D3D11SHADERMANAGER_H_

namespace ml {

	struct D3D11ShaderPair
	{
		D3D11PixelShader ps;
		D3D11GeometryShader gs;
		D3D11VertexShader vs;
	};

	class D3D11ShaderManager
	{
	public:
		D3D11ShaderManager()
		{
			m_graphics = NULL;
		}

		void init(GraphicsDevice& g);

		//! loads a vertex/pixel shader pair
		void registerShader(
			const std::string &filename,
			const std::string &shaderName,
			const std::string& entryPointVS = "vertexShaderMain",
			const std::string& shaderModelVS = "vs_4_0",
			const std::string& entryPointPS = "pixelShaderMain",
			const std::string& shaderModelPS = "ps_4_0",
			const std::vector<std::pair<std::string, std::string>>& shaderMacros = std::vector<std::pair<std::string, std::string>>());

		//! loads a vertex/geometry/pixel shader pair; set entryPointGS to "" for no geometry shader
		void registerShaderWithGS(
			const std::string &filename,
			const std::string &shaderName,
			const std::string& entryPointVS = "vertexShaderMain",
			const std::string& shaderModelVS = "vs_4_0",
			const std::string& entryPointGS = "geometryShaderMain",
			const std::string& shaderModelGS = "gs_4_0",
			const std::string& entryPointPS = "pixelShaderMain",
			const std::string& shaderModelPS = "ps_4_0",
			const std::vector<std::pair<std::string, std::string>>& shaderMacros = std::vector<std::pair<std::string, std::string>>());

		D3D11ShaderPair& getShaders(const std::string& shaderName)
		{
			MLIB_ASSERT_STR(m_shaders.count(shaderName) > 0, "shader not found in shader manager");
			return m_shaders.find(shaderName)->second;
		}

        const D3D11ShaderPair& getShaders(const std::string& shaderName) const
        {
            MLIB_ASSERT_STR(m_shaders.count(shaderName) > 0, "shader not found in shader manager");
            return m_shaders.find(shaderName)->second;
        }

		void bindShaders(const std::string& shaderName) const;
		void unbindShaders();
	private:
		D3D11GraphicsDevice* m_graphics;
		std::map<std::string, D3D11ShaderPair> m_shaders;
	};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11VERTEXSHADER_H_
