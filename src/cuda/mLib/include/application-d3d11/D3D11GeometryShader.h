
#ifndef APPLICATION_D3D11_D3D11GEOMETRYSHADER_H_
#define APPLICATION_D3D11_D3D11GEOMETRYSHADER_H_

namespace ml {

	class D3D11GeometryShader : public GraphicsAsset
	{
	public:
		D3D11GeometryShader()
		{
			m_shader = nullptr;
			m_blob = nullptr;
			m_graphics = nullptr;
		}

		~D3D11GeometryShader()
		{
			releaseGPU();
		}

		void init(
			GraphicsDevice& g, 
			const std::string& filename, 
			const std::string& entryPoint = "geometryShaderMain", 
			const std::string& shaderModel = "gs_4_0",
			const std::vector<std::pair<std::string, std::string>>& shaderMacros = std::vector<std::pair<std::string, std::string>>());

		void releaseGPU();
		void createGPU();

		void bind() const;

		bool isInit() const {
			return m_shader != nullptr;
		}
	private:
		D3D11GraphicsDevice *m_graphics;
		ID3D11GeometryShader *m_shader;
		ID3DBlob *m_blob;
		std::string m_filename;
	};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11GeometryShader_H_
