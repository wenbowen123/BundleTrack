#ifndef INCLUDE_CORE_NETWORK_NETWORKCLIENT_H_
#define INCLUDE_CORE_NETWORK_NETWORKCLIENT_H_

#ifdef _WIN32

#include <string>

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>


// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#define DEFAULT_BUFLEN 512

namespace ml {

	class NetworkClient
	{
	public:
		NetworkClient() {
			m_bIsOpen = false;
			m_serverSocket = INVALID_SOCKET;
		}
		~NetworkClient() {}

		bool open(const std::string& serverAddress, unsigned int port) {
			WSADATA wsaData;
			m_serverSocket = INVALID_SOCKET;
			struct addrinfo *result = NULL, *ptr = NULL, hints;
			int iResult;
			int recvbuflen = DEFAULT_BUFLEN;


			// Initialize Winsock
			iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
			if (iResult != 0) {
				printf("WSAStartup failed with error: %d\n", iResult);
				return false;
			}

			ZeroMemory(&hints, sizeof(hints));
			hints.ai_family = AF_UNSPEC;
			hints.ai_socktype = SOCK_STREAM;
			hints.ai_protocol = IPPROTO_TCP;

			// Resolve the server address and port
			iResult = getaddrinfo(serverAddress.c_str(), std::to_string(port).c_str(), &hints, &result);
			if (iResult != 0) {
				printf("getaddrinfo failed with error: %d\n", iResult);
				WSACleanup();
				return false;
			}

			// Attempt to connect to an address until one succeeds
			for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {

				// Create a SOCKET for connecting to server
				m_serverSocket = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
				if (m_serverSocket == INVALID_SOCKET) {
					printf("socket failed with error: %ld\n", WSAGetLastError());
					WSACleanup();
					return false;
				}

				// Connect to server.
				iResult = connect(m_serverSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
				if (iResult == SOCKET_ERROR) {
					closesocket(m_serverSocket);
					m_serverSocket = INVALID_SOCKET;
					continue;
				}
				break;
			}

			freeaddrinfo(result);

			if (m_serverSocket == INVALID_SOCKET) {
				printf("Unable to connect to server!\n");
				WSACleanup();
				return false;
			}


			m_bIsOpen = true;
			return true;
		}

		bool isOpen() const {
			return m_bIsOpen;
		}



		//! returns 0 if the connection was closed, the byte length, or -1 upon failure (non-blocking)
		int receiveData(BYTE* data, unsigned int bufferLen) {
			int iResult = recv(m_serverSocket, (char*)data, bufferLen, 0);
			return iResult;
		}

		//! returns 0 if the connection was closed, the byte length, or -1 upon failure (blocking function)
		int receiveDataBlocking(BYTE* data, unsigned int byteSize) {

			unsigned int bytesReceived = 0;
			if (m_bIsOpen) {
				while (bytesReceived < byteSize) {

					int size = receiveData(data + bytesReceived, byteSize - bytesReceived);
					if (size == -1)	return size;
					bytesReceived += size;
					std::cout << "total bytes: " << byteSize << std::endl;
					std::cout << "received bytes: " << bytesReceived << std::endl;
				}
			}
			else {
				throw MLIB_EXCEPTION("no connection with server");
			}
			return bytesReceived;
		}

		//! returns the number of send bytes if successfull; otherwise -1 (non-blocking function)
		int sendData(const BYTE* data, unsigned int byteSize) {

			int iSendResult = send(m_serverSocket, (const char*)data, byteSize, 0);
			if (iSendResult == SOCKET_ERROR) {
				printf("send failed with error: %d\n", WSAGetLastError());
				closesocket(m_serverSocket);
				WSACleanup();
				return SOCKET_ERROR;
			}
			return iSendResult;
		}

		//! blocks until all data is sent; returns the sent size upon success; -1 upon failure
		int sendDataBlocking(const BYTE* data, unsigned int byteSize) {
			int sentBytes = 0;
			while (sentBytes != byteSize) {

				int iResult = sendData(data + sentBytes, byteSize - sentBytes);
				if (iResult == SOCKET_ERROR)	return -1;
				sentBytes += iResult;
				std::cout << "total bytes: " << byteSize << std::endl;
				std::cout << "sent bytes: " << sentBytes << std::endl;
			}
			return sentBytes;
		}

		void close() {

			if (m_bIsOpen) {
				// shutdown the connection since no more data will be sent
				int iResult = shutdown(m_serverSocket, SD_SEND);
				if (iResult == SOCKET_ERROR) {
					printf("shutdown failed with error: %d\n", WSAGetLastError());
					closesocket(m_serverSocket);
					WSACleanup();
					//return false;
				}

				// cleanup
				closesocket(m_serverSocket);
				WSACleanup();
				m_bIsOpen = false;
			}
		}

	private:

		bool m_bIsOpen;
		SOCKET m_serverSocket;
	};

}  // namespace ml

#endif	// _WIN32

#endif  // INCLUDE_CORE_NETWORK_NETWORKCLIENT_H_


