
#ifndef APPLICATION_BASE_WINDOWWIN32_H_
#define APPLICATION_BASE_WINDOWWIN32_H_

#include <windows.h>

namespace ml {

	class ApplicationWin32;
	class WindowWin32 {
	public:
		WindowWin32(ApplicationWin32 &parent) :	m_parent(parent)
		{
			m_className = "uninitialized";

			m_handle = nullptr;
			msgProcCallback = nullptr;
			ZeroMemory(&m_class, sizeof(m_class));

			m_bResizeEvent = false;
		}
		~WindowWin32();

		typedef LRESULT (*MsgProcCallback)(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
		void init(HINSTANCE instance, int width, int height, const std::string &name, MsgProcCallback fun = nullptr, unsigned int initWindowPosX = 0, unsigned int initWindowPosY = 0);
		void destroy();
		void resize(UINT newWidth, UINT newHeight);
		void rename(const std::string &name);

		UINT getWidth() const;
		UINT getHeight() const;

		HWND getHandle() const
		{
			return m_handle;
		}
		ApplicationWin32& getParent()
		{
			return m_parent;
		}

		MsgProcCallback msgProcCallback;  // Called before messages are processed

	private:
		std::string			m_className;
		ApplicationWin32&	m_parent;
		WNDCLASS			m_class;
		HWND				m_handle;

		bool				m_bResizeEvent;
	};

}  // namespace ml

#endif  // APPLICATION_BASE_WINDOWWIN32_H_
