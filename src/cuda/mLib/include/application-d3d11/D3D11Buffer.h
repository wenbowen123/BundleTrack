
#ifndef APPLICATION_D3D11_D3D11BUFFER_H_
#define APPLICATION_D3D11_D3D11BUFFER_H_

namespace ml {

	template<class T>
	class D3D11Buffer : public GraphicsAsset
	{
	public:
		D3D11Buffer(bool _withSRV = true, bool _withUAV = false)	{
			m_graphics = nullptr;
			m_buffer = nullptr;
			m_srv = nullptr;
			m_uav = nullptr;
			
			m_bHasSRV = _withSRV;
			m_bHasUAV = _withUAV;
		}

		D3D11Buffer(GraphicsDevice& g, const std::vector<T>& data, bool _withSRV = true, bool _withUAV = false)	{
			m_graphics = nullptr;
			m_buffer = nullptr;
			m_srv = nullptr;
			m_uav = nullptr;

			m_bHasSRV = _withSRV;
			m_bHasUAV = _withUAV;
			init(g, data);
		}

		//! copy constructor
		D3D11Buffer(D3D11Buffer& t) {
			m_graphics = nullptr;
			m_buffer = nullptr;
			m_srv = nullptr;
			m_uav = nullptr;

			m_bHasSRV = t.m_bHasSRV;
			m_bHasUAV = t.m_bHasUAV;
			init(*t.m_graphics, t.getData());
		}

		//! move constructor
		D3D11Buffer(D3D11Buffer&& t)	{
			m_graphics = nullptr;
			m_buffer = nullptr;
			m_srv = nullptr;
			m_uav = nullptr;

			m_bHasSRV = false;
			m_bHasUAV = false;

			swap(*this, t);
		}

		~D3D11Buffer()	{
			releaseGPU();
		}

		//! assignment operator
		void operator=(D3D11Buffer& t) {
			m_bHasSRV = t.m_bHasSRV;
			m_bHasUAV = t.m_bHasUAV;
			init(*t.m_graphics, t.getData());
		}

		//! move operator
		void operator=(D3D11Buffer&& t)	{
			swap(*this, grid);
		}				

		//! adl swap
		friend void swap(D3D11Buffer& a, D3D11Buffer& b) {
			std::swap(a.m_graphics, b.m_graphics);
			std::swap(a.m_data, b.m_data);
			std::swap(a.m_buffer, b.m_buffer);
			std::swap(a.m_srv, b.m_srv);
			std::swap(a.m_uav, b.m_uav);

			std::swap(a.m_bHasSRV, b.m_bHasSRV);
			std::swap(a.m_bHasUAV, b.m_bHasUAV);
		}

		void init(GraphicsDevice& g, const std::vector<T>& data);
		void releaseGPU();
		void createGPU();

		//! binds the buffer as a shader resource view
		void bindSRV(unsigned int slot = 0) const;
		//! unbinds the shader resource view
		void unbindSRV(unsigned int slot = 0) const;

		const std::vector<T>& getData() const {
			return m_data;
		}

		bool hasSRV() const {
			return m_bHasSRV;
		}
		bool hasUAV() const {
			return m_bHasUAV;
		}

		bool isInit() const {
			return m_buffer != nullptr;
		}

	private:
		D3D11GraphicsDevice* m_graphics;
		std::vector<T> m_data;
		ID3D11Buffer* m_buffer;
		ID3D11ShaderResourceView* m_srv;
		ID3D11UnorderedAccessView* m_uav;

		bool m_bHasSRV;
		bool m_bHasUAV;
	};

}  // namespace ml

#include "D3D11Buffer.cpp"

#endif  // APPLICATION_D3D11_D3D11Buffer_H_
