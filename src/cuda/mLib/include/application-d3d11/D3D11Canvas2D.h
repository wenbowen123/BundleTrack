
#ifndef APPLICATION_D3D11_D3D11CANVAS2D_H_
#define APPLICATION_D3D11_D3D11CANVAS2D_H_

namespace ml {

	class D3D11Canvas2D : public GraphicsAsset
	{
	public:
		struct Intersection
		{
			Intersection()
			{
				coord.x = coord.y = std::numeric_limits<int>::min();
			}
			bool isValid() const
			{
				return (coord.x != std::numeric_limits<int>::min());
			}
			Intersection(const std::string &id, const vec2i &_coord) : elementId(id), coord(_coord) {}
			std::string elementId;
			vec2i coord;
		};

		enum ElementType
		{
			ELEMENT_TYPE_BILLBOARD,
			ELEMENT_TYPE_CIRCLE,
			ELEMENT_TYPE_BOX
		};

		
		//forward declarations for casts
		class ElementBillboard;
		class ElementCircle;
		class ElementBox;

		class Element
		{
		public:
			Element(GraphicsDevice& g, const std::string &id, ElementType elementType, float depth, bool useDefaultShader = true) 
				: m_graphics(&g.castD3D11()), m_id(id), m_depth(depth), m_bUseDefaultShader(useDefaultShader) {
				m_elementType = elementType;
			}

			virtual ~Element() {}

			//! render the element: takes care for shader binding, etc.,
			virtual void render() = 0;
			virtual void resize() {};
			virtual bool intersects(const vec2i &mouseCoord, Intersection &intersection) const
			{
				return false;
			}

			ElementType getType() const {
				return m_elementType;
			}

			const ElementBillboard& castBillboard() const {
				MLIB_ASSERT(m_elementType == ELEMENT_TYPE_BILLBOARD);
				return *((ElementBillboard*)this);
			}
			ElementBillboard& castBillboard() {
				MLIB_ASSERT(m_elementType == ELEMENT_TYPE_BILLBOARD);
				return *((ElementBillboard*)this);
			}

			const ElementCircle& castCircle() const {
				MLIB_ASSERT(m_elementType == ELEMENT_TYPE_CIRCLE);
				return *((ElementCircle*)this);
			}
			ElementCircle& castCircle() {
				MLIB_ASSERT(m_elementType == ELEMENT_TYPE_CIRCLE);
				return *((ElementCircle*)this);
			}

			const ElementBox& castBox() const {
				MLIB_ASSERT(m_elementType == ELEMENT_TYPE_BOX);
				return *((ElementBox*)this);
			}
			ElementBox& castBox() {
				MLIB_ASSERT(m_elementType == ELEMENT_TYPE_BOX);
				return *((ElementBox*)this);
			}


			friend D3D11Canvas2D;
		protected:
			D3D11GraphicsDevice* m_graphics;
			float m_depth; // value should be between 0 and 1
			std::string m_id;
			bool m_bUseDefaultShader;

		private:
			ElementType m_elementType;
		};

		class ElementBillboard : public Element
		{
		public:
			ElementBillboard(GraphicsDevice& g, const std::string &id, const bbox2i& box, const ColorImageR8G8B8A8& image, float depth, bool useDefaultShader) : Element(g, id, ELEMENT_TYPE_BILLBOARD, depth, useDefaultShader)
			{
				const std::string mLibShaderDir = util::getMLibDir() + "data/shaders/";
				m_graphics->getShaderManager().registerShader(mLibShaderDir + "defaultCanvas.hlsl", "defaultCanvasBillboard", "billboardVS", "vs_4_0", "billboardPS", "ps_4_0");

				m_id = id;
				m_box = box;
				m_tex.init(g, image);
				resize();
			}

			void resize();
			void render();
			bool intersects(const vec2i &mouseCoord, Intersection& intersection) const;

			const bbox2i& getBox() const
			{
				return m_box;
			}

			void updateTexture(const ColorImageR8G8B8A8& image) {
				m_tex.init(*m_graphics, image);
			}

			friend D3D11Canvas2D;
		private:
			bbox2i m_box;
			D3D11Texture2D<vec4uc> m_tex;
			D3D11TriMesh m_mesh;
		};

		class ElementCircle : public Element
		{
		public:
			struct ElementCircleConstants {
				vec4f color;
				vec2f center;
				float radius;
				float dummy;
			};
			ElementCircle(GraphicsDevice& g, const std::string &id, const vec2f& center, float radius, const vec4f& color, float depth) : Element(g, id, ELEMENT_TYPE_CIRCLE, depth) {

				const std::string mLibShaderDir = util::getMLibDir() + "data/shaders/";
				m_graphics->getShaderManager().registerShader(mLibShaderDir + "defaultCanvas.hlsl", "defaultCanvasCircle", "circleVS", "vs_4_0", "circlePS", "ps_4_0");

				m_constants.center = center;
				m_constants.radius = radius;
				m_constants.color = color;
				m_constantBuffer.init(g);
				m_constantBuffer.update(m_constants);

				resize();
			}

			void resize() {
				bbox2f box;
				box.include(m_graphics->pixelToNDC(math::floor(m_constants.center - m_constants.radius)));
				box.include(m_graphics->pixelToNDC(math::ceil(m_constants.center + m_constants.radius)));
				m_mesh.init(*m_graphics, ml::Shapesf::rectangleZ(box.getMin(), box.getMax(), m_depth));
			}

			void render() {
				if (m_bUseDefaultShader) m_graphics->getShaderManager().bindShaders("defaultCanvasCircle");
				m_constantBuffer.bind(0);
				m_mesh.render();
			}

			friend D3D11Canvas2D;
		private:
			ElementCircleConstants m_constants;
			D3D11ConstantBuffer<ElementCircleConstants> m_constantBuffer;
			D3D11TriMesh m_mesh;
		};

		class ElementBox : public Element
		{
		public:
			ElementBox(GraphicsDevice& g, const std::string &id, const bbox2i& box, const vec4f& color, float depth, bool useDefaultShader) 
				: Element(g, id, ELEMENT_TYPE_BOX, depth, useDefaultShader)
			{
				const std::string mLibShaderDir = util::getMLibDir() + "data/shaders/";
				m_graphics->getShaderManager().registerShader(mLibShaderDir + "defaultCanvas.hlsl", "defaultCanvasBox", "boxVS", "vs_4_0", "boxPS", "ps_4_0");
				m_id = id;
				m_box = box;
				m_color = color;
				resize();
			}

			void resize() {
				bbox2f boxNdc;
				boxNdc.include(m_graphics->pixelToNDC(m_box.getMin()));
				boxNdc.include(m_graphics->pixelToNDC(m_box.getMax()));
				m_mesh.init(*m_graphics, ml::Shapesf::rectangleZ(boxNdc.getMin(), boxNdc.getMax(), m_depth, m_color));
			}

			void render() {
				if (m_bUseDefaultShader)  m_graphics->getShaderManager().bindShaders("defaultCanvasBox");
				m_mesh.render();
			}
			bool intersects(const vec2i &mouseCoord, Intersection &intersection) const {
				if (m_box.intersects(mouseCoord))
				{
					intersection = D3D11Canvas2D::Intersection(m_id, mouseCoord - m_box.getMin());
					return true;
				}
				return false;
			}

			const bbox2i& getBox() const
			{
				return m_box;
			}

			friend D3D11Canvas2D;
		private:
			bbox2i m_box;
			D3D11TriMesh m_mesh;
			vec4f m_color;
		};


		D3D11Canvas2D() {
			m_graphics = nullptr;
		}
		//! copy constructor
		D3D11Canvas2D(D3D11Canvas2D& other) {
			m_graphics = nullptr;
			init(*other.m_graphics);
			for (auto &e : other.m_namedElements)
				addElement(e.first, *e.second);
			for (Element* e : other.m_unnamedElements)
				addElement(*e);
		}
		//! move constructor
		D3D11Canvas2D(D3D11Canvas2D&& other) {
			m_graphics = nullptr;
			swap(*this, other);
		}

		~D3D11Canvas2D() {
			clearElements();
		}

		//! adl swap
		friend void swap(D3D11Canvas2D& a, D3D11Canvas2D& b) {
			std::swap(a.m_graphics, b.m_graphics);
			std::swap(a.m_namedElements, b.m_namedElements);
			std::swap(a.m_unnamedElements, b.m_unnamedElements);
		}
		
		//! assignment operator
		void operator=(D3D11Canvas2D& other) {
			if (this != &other) {
				init(*other.m_graphics);
				for (auto &e : other.m_namedElements)
					addElement(e.first, *e.second);
				for (Element* e : other.m_unnamedElements)
					addElement(*e);
			}
		}
		//! move operator
		void operator=(D3D11Canvas2D&& other) {
			swap(*this, other);
		}
			 
		void init(GraphicsDevice& g);

		void addCircle(const std::string& elementId, const vec2f& centerInPixels, float radiusInPixels, const vec4f& color, float depth) {
			if (elementId == "")	addCircle(centerInPixels, radiusInPixels, color, depth);
			else
				m_namedElements[elementId] = new ElementCircle(*m_graphics, elementId, centerInPixels, radiusInPixels, color, depth);
		}

		void addBillboard(const std::string& elementId, const bbox2i& box, const ColorImageR8G8B8A8 &image, float depth, bool useDefaultShader = true) {
			if (elementId == "")	addBillboard(box, image, depth, useDefaultShader);
			else
				m_namedElements[elementId] = new ElementBillboard(*m_graphics, elementId, box, image, depth, useDefaultShader);
		}
		void addBox(const std::string& elementId, const bbox2i& box, const vec4f& color, float depth, bool useDefaultShader = true) {
			if (elementId == "")	addBox(box, color, depth, useDefaultShader);
			else
				m_namedElements[elementId] = new ElementBox(*m_graphics, elementId, box, color, depth, useDefaultShader);
		}

		void addCircle(const vec2f& centerInPixels, float radiusInPixels, const vec4f& color, float depth) {
			m_unnamedElements.push_back(new ElementCircle(*m_graphics, "", centerInPixels, radiusInPixels, color, depth));
		}

		void addBillboard(const bbox2i& box, const ColorImageR8G8B8A8& image, float depth, bool useDefaultShader = true) {
			m_unnamedElements.push_back(new ElementBillboard(*m_graphics, "", box, image, depth, useDefaultShader));
		}

		void addBox(const bbox2i& box, const vec4f& color, float depth, bool useDefaultShader = true) {
			m_unnamedElements.push_back(new ElementBox(*m_graphics, "", box, color, depth, useDefaultShader));
		}

		//! for copy operators
		void addElement(const std::string& elementId, const Element& e) {
			auto type = e.getType();
			if (type == ELEMENT_TYPE_BILLBOARD) {
				auto _e = e.castBillboard();
				addBillboard(elementId, _e.m_box, ColorImageR8G8B8A8(_e.m_tex.getImage()), _e.m_depth);
			}
			else if (type == ELEMENT_TYPE_BOX) {
				auto _e = e.castBox();
				addBox(elementId, _e.m_box, _e.m_color, _e.m_depth, _e.m_bUseDefaultShader);
			}
			else if (type == ELEMENT_TYPE_CIRCLE) {
				auto _e = e.castCircle();
				addCircle(elementId, _e.m_constants.center, _e.m_constants.radius, _e.m_constants.color, _e.m_depth);
			}
			else throw MLIB_EXCEPTION("unknown type");
		}
		void addElement(const Element& e) {
			addElement("", e);
		}

		Intersection intersectionFirst(const vec2i& mouseCoord) const;
		std::vector<Intersection> intersectionAll(const vec2i& mouseCoord) const;

		void releaseGPU();
		void createGPU();
		void resize();

		void render();
		void render(const std::string& elementId);

		void clearElements() {
			for (Element *e : m_unnamedElements)
				SAFE_DELETE(e);
			for (auto &e : m_namedElements)
				SAFE_DELETE(e.second);
			m_namedElements.clear();
			m_unnamedElements.clear();
		}

		bool elementExists(const std::string& elementId) const {
			auto it = m_namedElements.find(elementId);
			return it != m_namedElements.end();
		}

		Element& getElementById(const std::string &elementId) {
			auto it = m_namedElements.find(elementId);
			MLIB_ASSERT_STR(it != m_namedElements.end(), "Element not found");
			return *(it->second);
		}

	private:
		D3D11GraphicsDevice* m_graphics;
		std::map<std::string, Element*> m_namedElements;
		std::vector<Element*> m_unnamedElements;
	};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11CANVAS2D_H_
