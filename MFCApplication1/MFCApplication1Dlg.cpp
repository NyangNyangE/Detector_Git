﻿
// MFCApplication1Dlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "MFCApplication1.h"
#include "MFCApplication1Dlg.h"
#include "afxdialogex.h"
#include <afxwin.h>

//여기부터
#include <stdio.h>
#include "include/opencv/opencv2/opencv.hpp"
#include "include/yolo/yolo_v2_class.hpp"

#include <stdlib.h>
#include <string.h>
#include <atlstr.h>
#include <iostream>

#include <WinInet.h>
#include <WinSock2.h>
#include <winsock.h>
#include <Windows.h>
#include <WS2tcpip.h>

#include <cstdio>
#include <future>

#include <iostream>
#include <fstream>
#include <io.h>
#include <conio.h>
#include <ctime>
#include <thread>
#include <time.h>
#include <mutex>
#include <algorithm>
#include <DbgHelp.h>

#pragma comment(lib, "dbghelp.lib")

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define PIPE_NAME "c++_pipe"



using namespace std;
using namespace cv;

DWORD threadId;
HANDLE hThread = NULL;

unsigned long __stdcall NET_RvThr(void * pParam);
HANDLE hPipe1, hPipe2;
BOOL Finished;

LPTSTR lpszPipename1 = TEXT("\\\\.\\pipe\\c++_send"); // 쓰기  
LPTSTR lpszPipename2 = TEXT("\\\\.\\pipe\\c#_send");  // 읽기



//Detector detector2("C:\\Program Files (x86)\\Twowinscom\\Parking Guidance\\Detector\\yolov3_c10.cfg", "C:\\Program Files (x86)\\Twowinscom\\Parking Guidance\\Detector\\yolov3_c10_10000.weights");
//Detector detector2("yolov3_c10.cfg", "yolov3_c10_10000.weights");


/* 변수 및 구조체 */

typedef struct receiveData {											//roi 영역 txt 파일을 읽어 저장
	string fileName;													//파일 이름
	string panicAddr;
	string ip;
	int cameraNum;
	int x[3];															//roi x 좌표 3개
	int y[3];															//    y 좌표 3개
	int areas;                                                          //주차면 갯수
	bool isRed;
	int cars;															//주차 차량수
	Mat image;
	bool isDown;
	string numbers[3];                       							//번호 판별 결과 저장
	receiveData() : numbers{ "x", "x", "x" }, isRed(false), isDown(false) {}
}receiveData;


typedef struct plateData {												//찾은 번호판 영역좌표 및 기타정보
	string fileName;
	string panicName;
	string dvrIP;
	string plateNum[3];
	int cameraNum;
	int x[3], y[3], w[3], h[3];											//찾은 번호판 좌표
	int cPlate;													     	//찾은 번호판 개수
	bool isRed;											         		//사진에서 만차 판별 결과 ( c# UI 에서 2장 모두 red 일때 red로 변경)
	plateData() : plateNum{"x", "x", "x"}, cPlate(0), isRed(false) {}
}plateData;


const int READ_BUF_SIZE = 1920 * 1080;
HWND g_hMain;															//UI handle
HWND handle;
//vector<receiveData> readList;											//분석 해야될 영상 리스트
vector<plateData> plateList;											//분석 결과 리스트
map<int, receiveData> dataList;
Mat original[4];														//번호 인식에 사용되는 원본 영상 저장하는 리스트
std::mutex mtx;															//plateList 분석된 결과 쓰기 읽기, 이미지 쓰기 읽기 동시 진행 방지
bool detected = false;													//프로그램 시작시 번호 인식 한번 실행, 다음부턴 UI 에서 신호 있을때만 실행
Mat default_image(540, 960, CV_8UC3);									//4장씩 영상 합칠때 영상 없을시 사용되는 아무것도 없는 이미지


int ccc = 0; //test


void TcpSend(bool isred);													//만공등 변경 현재 사용 X
void getHttpPath();															//txt 파일읽어서 readList 값채우고 
int getFileFromHttp(string pszUrl, int index);							//이미지 다운로드
void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec);			//사각형 그리기 현재 사용 X
bool cmp(const bbox_t& b1, const bbox_t& b2);								//비교
void send();																//차량 유무 red, green  UI 에 보냄
void send2();																//차량 유무 + 번호      UI 에 보냄
void detectNumber();														//번호 인식
vector<string> split2(string str, char delimiter);							//문자열 나누기 split 
void ErrorHandling(const char message[]);									//에러출력

void self_mini_dump();														//예외 발생시 덤프 파일 남김
LONG WINAPI top_level_filter(__in PEXCEPTION_POINTERS pExceptionPointer);   //덤프 파일
string get_dump_filename();													//덤프 파일 명
string getTime();															//현재시간 ( 에러 메시지에 표시 ) 
void writeErrorMessage(string msg);											//에러메시지 텍스트 파일 출력


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};



CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFCApplication1Dlg 대화 상자



CMFCApplication1Dlg::CMFCApplication1Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_MFCAPPLICATION1_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMFCApplication1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CMFCApplication1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CMFCApplication1Dlg::OnBnClickedOk)
	ON_WM_COPYDATA()
	ON_WM_WINDOWPOSCHANGING()
END_MESSAGE_MAP()


// CMFCApplication1Dlg 메시지 처리기

BOOL CMFCApplication1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.


	SetWindowText(_T("Detector"));

	g_hMain = ::FindWindow(NULL, _T("PanicManagementSystem"));

	if (g_hMain != NULL)
	{
		handle = GetSafeHwnd();
	}
	else
	{
		AfxMessageBox(_T("메인UI실행안됨"));
		exit(0);
	}

	default_image = Scalar(0); //이미지 없을시 기본 흑백 이미지를 사용 image is not open error 처리하기 위해

	PipeOpen();

	self_mini_dump();          //현재 시점부터 예외 발생시 덤프 파일 남김

	ofstream outFile("ExceptionLog.txt", ios::app); // 로그 파일 생성
	outFile << getTime() << "Program Start" << endl;
	outFile.close();

	CreateThread(NULL, 0, &ThreadProc, NULL, 0, NULL);	//detect 판별 시작

	ShowWindow(SW_SHOWMINIMIZED);//! 최소화후 숨겨야 화면에 창 나타나지 않음
	PostMessage(WM_SHOWWINDOW, FALSE, SW_OTHERUNZOOM);
	//SetBackgroundColor(RGB(200, 200, 200));
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CMFCApplication1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CMFCApplication1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CMFCApplication1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}











/* 덤프 파일 */
void self_mini_dump()
{
	SetUnhandledExceptionFilter(top_level_filter);										//예외시 이벤트 등록
}
LONG WINAPI top_level_filter(__in PEXCEPTION_POINTERS pExceptionPointer)
{
	MINIDUMP_EXCEPTION_INFORMATION MinidumpExceptionInformation;
	std::string dump_filename;

	MinidumpExceptionInformation.ThreadId = ::GetCurrentThreadId();
	MinidumpExceptionInformation.ExceptionPointers = pExceptionPointer;
	MinidumpExceptionInformation.ClientPointers = FALSE;

	std::wstring dump;

	dump_filename = get_dump_filename();
	std::wstring filename = std::wstring(dump_filename.begin(), dump_filename.end());
	if (dump_filename.empty() == true)
	{
		::TerminateProcess(::GetCurrentProcess(), 0);

		writeErrorMessage(" dump_filename is not exist ");
	}

	HANDLE hDumpFile = ::CreateFileW(filename.c_str(),
		GENERIC_WRITE,
		FILE_SHARE_WRITE,
		NULL,
		CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, NULL);

	MiniDumpWriteDump(GetCurrentProcess(),
		GetCurrentProcessId(),
		hDumpFile,
		MiniDumpNormal,
		&MinidumpExceptionInformation,
		NULL,
		NULL);
	::TerminateProcess(::GetCurrentProcess(), 0);

	return 0;
}
string get_dump_filename()
{
	time_t rawtime;
	struct tm timeinfo;

	std::string date_string;

	std::wstring module_path;
	string dump_filename;

	static WCHAR ModulePath[1024];

	time(&rawtime);
	localtime_s(&timeinfo, &rawtime);

	//1900년 기준시작
	date_string = to_string(timeinfo.tm_year + 1900) + to_string(timeinfo.tm_mon + 1) +
		to_string(timeinfo.tm_mday) + to_string(timeinfo.tm_hour) + to_string(timeinfo.tm_min) +
		to_string(timeinfo.tm_sec);



	if (::GetModuleFileNameW(0, ModulePath, sizeof(ModulePath) / sizeof(WCHAR)) == 0)
	{
		return std::string();
	}

	module_path = ModulePath;

	dump_filename = ".\\" + date_string + ".dmp";

	return dump_filename;
}


void CMFCApplication1Dlg::PipeOpen()
{
	try
	{
		Finished = FALSE;

		hPipe1 = CreateFile(lpszPipename1, GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL); //쓰기 c++ -> c#
		hPipe2 = CreateFile(lpszPipename2, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL); //읽기

		if ((hPipe1 == NULL || hPipe1 == INVALID_HANDLE_VALUE) || (hPipe2 == NULL || hPipe2 == INVALID_HANDLE_VALUE))
		{
			TRACE("Could not open the pipe  - (error %d)\n", GetLastError());
		}
		else
		{
			hThread = CreateThread(NULL, 0, &NET_RvThr, NULL, 0, NULL);
		}
	}
	catch (exception ex)
	{
		writeErrorMessage(to_string(__LINE__) + " PipeOpen error :" + (string)ex.what());
	}
}


unsigned long __stdcall NET_RvThr(void * pParam)						// UI 에서 받은 메시지 처리
{
	BOOL fSuccess;
	char chBuf[100];
	DWORD dwBytesToWrite = (DWORD)100;
	DWORD cbRead;
	int i;
	string receive;

	while (true)
	{
		receive = "";
		fSuccess = ReadFile(hPipe2, chBuf, dwBytesToWrite, &cbRead, NULL);
		if (fSuccess)
		{
			//receive = chBuf;

			//if (receive._Equal("search"))				//UI 에서 search 받을때만 번호인식  ( 현재 UI 에서 1분에 한번으로 설정해놓음 -> 변경예정)
			//{
			//	
			//}
		}
		if (!fSuccess && GetLastError() != ERROR_MORE_DATA)
		{
			break;
		}
	}

	return 0;
}



DWORD WINAPI CMFCApplication1Dlg::ThreadProc(void *p)
{
	CMFCApplication1Dlg *CIPCS = (CMFCApplication1Dlg*)p;
	CIPCS->detect();
	return 0;
}






void TcpSend(bool isred)	//사용 X    UI 에서 처리
{

	WSADATA     wsaData;
	SOCKET      hSocket;
	SOCKADDR_IN servAddr;

	int     port = 5000;

	BYTE bRed[] = { 0x02, 0x21, 0x30, 0x30, 0x32, 0xB5, 0x03 }; //22
	BYTE bGreen[] = { 0x02, 0x20, 0x30, 0x30, 0x32, 0xB4, 0x03 };

	const char *p = NULL;
	if (isred)
	{
		p = reinterpret_cast<const char*>(bRed);
	}
	else if (isred == false)
	{
		p = reinterpret_cast<const char*>(bGreen);
	}

	int     strLen;

	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
		ErrorHandling("WSAStartup() error!");

	hSocket = socket(PF_INET, SOCK_STREAM, 0);

	if (hSocket == INVALID_SOCKET)
		ErrorHandling("hSocket() error!");

	memset(&servAddr, 0, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(port);
	inet_pton(AF_INET, "192.168.0.74", &servAddr.sin_addr.s_addr);

	if (connect(hSocket, (SOCKADDR*)&servAddr, sizeof(servAddr)) == SOCKET_ERROR)
		ErrorHandling("connect() error!");

	send(hSocket, p, sizeof(p), 0);

	closesocket(hSocket);
	WSACleanup();

}

void ErrorHandling(const char message[])
{
	fputs(message, stderr);
	fputc('\n', stderr);
}


void downImage()
{
	CreateDirectoryW(_T(".\\downImage"), NULL);
	getHttpPath();
}

vector<string> split2(string str, char delimiter) {
	vector<string> internal;

	try
	{
		stringstream ss(str);
		string temp;

		while (getline(ss, temp, delimiter)) {
			internal.push_back(temp);
		}
	}
	catch (exception ex)
	{
		writeErrorMessage(to_string(__LINE__) + " split2 error :" + (string)ex.what());
	}

	return internal;
}


void getHttpPath()
{

	std::string pathName = "C:\\Program Files (x86)\\Twowinscom\\Parking Guidance\\downImage";  //파일 저장 위치						

	ifstream in("C:\\Program Files (x86)\\Twowinscom\\Parking Guidance\\roi.txt");		//roi 영역 및 카레마 번호등 정보 txt
	//ifstream in("D:\\새 폴더\\차량번호\\차량번호U\\PanicCall\\bin\\x86\\Release\\roi.txt");

	string line;
	vector<string> data;
	vector<std::future<int>> pending;
	int idx = 0;

	while (getline(in, line))			//한라인씩 읽어서 비동기 작업으로 다운로드 처리
	{
		try {
			data = split2(line, ' ');
			string url = "http://admin:admin@" + data[0] + "/still.jpg";	//url 변경
			string path = pathName;
			path += "\\" + data[1] + "_" + data[0] + ".jpg";

			//getFileFromHttp(url, idx);

			auto th = async(launch::async, getFileFromHttp, url, idx);		//path = 저장될 파일 명
			pending.push_back(std::move(th));


			//
			dataList[idx].panicAddr = data[1];
			dataList[idx].fileName = data[1] + "_" + data[0];
			dataList[idx].ip = data[0];


			int index = 0, count = 0;
			for (int i = 2; i < data.size(); i += 2)						//roi 정보 저장 (x y), (x y), (x y) 최대 3개 roi 없을시 (0,0) 
			{
				dataList[idx].x[index] = stoi(data[i]);
				dataList[idx].y[index] = stoi(data[i + 1]);
				if (dataList[idx].x[index] != 0 || dataList[idx].y[index] != 0)
					count++;
				index++;
			}


			dataList[idx].areas = count;
			idx++;
			Sleep(10);//tes
		}
		catch (exception ex) {
			writeErrorMessage(to_string(__LINE__) + " download image or readList data error :" + (string)ex.what());
		}
		catch (cv::Exception& e) //opencv 예외
		{
			writeErrorMessage(to_string(__LINE__) + " getHttpPath opencv error :" + (string)e.what());
		}
	}

	in.close();

	try {
		for (int i = 0; i < pending.size(); i++)					//wait async thread  : 다운로드 완료 되기전 detect() 에서image 읽기 시도 방지
		{
			//pending.at(i).wait_for(chrono::milliseconds(15));
			pending.at(i).get();
		}
	}
	catch (exception ex)
	{
	}


}

//"http://admin:tec15885772!@192.168.0.206/cgi-bin/sdk/video_sdk.cgi?msubmenu=jpg&ch=0";
//정상적인 루틴 완료 후 재인식 과정 실행.
int getFileFromHttp(string pszUrl, int index)
{
	HINTERNET    hInet, hUrl;
	DWORD        dwReadSize = 0;
	BYTE	*Data; 
	unique_ptr<BYTE[]> rawData(new BYTE[READ_BUF_SIZE]);

	try
	{
		std::wstring temp = std::wstring(pszUrl.begin(), pszUrl.end()); //std::string -> LPCWSTR
		LPCWSTR wstr = temp.c_str();

		// WinINet함수 초기화

		if ((hInet = InternetOpen(_T("Myweb"),            // user agent in the HTTP protocol
			INTERNET_OPEN_TYPE_DIRECT,    // AccessType
			NULL,                        // ProxyName
			NULL,                        // ProxyBypass
			0)) != NULL)                // Options
		{
			// 입력된 HTTP주소를 열기
			DWORD dwTimeout = 100;
			InternetSetOption(hInet, INTERNET_OPTION_SEND_TIMEOUT, &dwTimeout, sizeof(DWORD));
			InternetSetOption(hInet, INTERNET_OPTION_RECEIVE_TIMEOUT, &dwTimeout, sizeof(DWORD));
			InternetSetOption(hInet, INTERNET_OPTION_CONNECT_TIMEOUT, &dwTimeout, sizeof(DWORD));
			InternetSetOption(hInet, INTERNET_OPTION_CONTROL_RECEIVE_TIMEOUT, &dwTimeout, sizeof(DWORD));
			InternetSetOption(hInet, INTERNET_OPTION_CONTROL_SEND_TIMEOUT, &dwTimeout, sizeof(DWORD));
			InternetSetOption(hInet, INTERNET_OPTION_DATA_SEND_TIMEOUT, &dwTimeout, sizeof(DWORD));
			InternetSetOption(hInet, INTERNET_OPTION_DATA_RECEIVE_TIMEOUT, &dwTimeout, sizeof(DWORD));

			if ((hUrl = InternetOpenUrl(hInet,        // 인터넷 세션의 핸들
				wstr,                        // URL
				NULL,                        // HTTP server에 보내는 해더
				0,                           // 해더 사이즈
				INTERNET_FLAG_RELOAD,        // Flag
				0)) != NULL)                 // Context
			{

				
				DWORD	 dwSize;			//읽어 오는 길이
				DWORD    dwRead;			//실제 읽은 길이
				DWORD    dwDebug = 3;
				DWORD	 dwIndex = 0;
				int total = 0;

				do
				{
					//한번에 가져올 수 있는 파일 길이
					InternetQueryDataAvailable(hUrl, &dwSize, 0, 0);

					// 웹상의 파일 읽기
					BOOL bRet = InternetReadFile(hUrl, &rawData[dwIndex], dwSize, &dwRead);
					//BOOL bRet = InternetReadFile(hUrl, rawData, READ_BUF_SIZE, &dwRead);

					if (bRet) {
						dwIndex += dwRead; total += dwRead;
					}

				} while (dwRead != 0 || dwDebug-- > 0);

				//JPG 타입 검사
				if (rawData[0] == 0xFF && rawData[1] == 0xD8 && rawData[total - 2] == 0xFF && rawData[total - 1] == 0xD9)
				{
					Mat raw(1080, 1920, CV_8UC3, rawData.get());
					Mat decodedImage = imdecode(raw, CV_LOAD_IMAGE_COLOR);
					//resize(decodedImage, imageList[index], cv::Size(960, 540)); 오류코드 (나중에 테스트)
					dataList[index].image = decodedImage;
					dataList[index].isDown = true;

					raw.release();
					//decodedImage.release();
				}

				//인터넷 핸들 닫기
				InternetCloseHandle(hUrl);
			}
			else
			{
				cout << "Internet Connection failed :" << dataList[index].ip << endl;
			}

			// 인터넷 핸들 닫기
			InternetCloseHandle(hInet);

		}

	}
	catch (exception ex) //일반 예외
	{
		writeErrorMessage(to_string(__LINE__) + " getFileFromHttp error :" + (string)ex.what());
	}
	catch (cv::Exception& e) //opencv 예외
	{
		writeErrorMessage(to_string(__LINE__) + " getFileFromHttp opencv error :" + (string)e.what());
	}

	//delete[] rawData;

	return (int)dwReadSize;

	/*std::wstring url = std::wstring(pszUrl.begin(), pszUrl.end());
	std::wstring file = std::wstring(pszFile.begin(), pszFile.end());
	URLDownloadToFile(
	URLDownloadToFile(NULL, url.c_str(), file.c_str(), NULL, NULL);
	return 1;*/
}


//다운로드 완료된 이미지들 4장으로 합친 새로운 이미지 생성 (1920 x 1080 )
//마지막에 남은 이미지가 4장이 아니고 3장과 같은 경우  나머지 한칸 default image 로 채워짐
Mat concatImg(int index)
{
	// 4장의 이미지가 하나의 1920 1080  이미지의 각각의 자리로 복사됨
	// 1번이미지 0,0 부터 시작
	// 2번이미지 960, 0
	// 3번이미지 0, 960
	// 4번이미지 960, 960

	Mat resultMat(1080, 1920, CV_8UC3);
	vector<Mat> imgArray(4);
	Mat img0, img1, img2, img3;


	//기본 4장 단위로 합치므로 4장이 아닐시 처리를 위한 size 설정 size 는 0 ~ 4사이의 값을 가짐
	int size = 4;
	try {
		if ((dataList.size() - index) < 4)
			size = dataList.size() - index;
	}
	catch (Exception ex)
	{
		cout << "readList index error " << ex.what() << endl;
		writeErrorMessage(to_string(__LINE__) + "  concat image readList null error :" + (string)ex.what());
	}



	//합쳐질 4장의 이미지 불러오기  각각의 스레드 총 4개
	//불러온 이미지 1/4 크기로 변경

	thread t[4], t1, t2, t3, t4;
	for (int i = 0; i < size; i++)
	{
		t[i] = thread([](int i)
		{
			try
			{
				if (dataList[i].image.empty())
					dataList[i].image = default_image;
				else
					resize(dataList[i].image, dataList[i].image, cv::Size(960, 540));
			}
			catch (std::exception e) {
				dataList[i].image = default_image;
			}
			catch (cv::Exception& e) //opencv 예외
			{
			}
		}, i + index);
	}

	try
	{
		// 각각의 자리로 복사 
		for (int i = 0; i < size; i++)
		{
			t[i].join();

			switch (i)
			{
			case 0:
				dataList[i + index].image.copyTo(resultMat(cv::Rect(0, 0, 960, 540)));
				dataList[i + index].image.release(); 
				break;
			case 1:
				dataList[i + index].image.copyTo(resultMat(cv::Rect(960, 0, 960, 540)));
				dataList[i + index].image.release();
				break;
			case 2:
				dataList[i + index].image.copyTo(resultMat(cv::Rect(0, 540, 960, 540)));
				dataList[i + index].image.release();
				break;
			case 3:
				dataList[i + index].image.copyTo(resultMat(cv::Rect(960, 540, 960, 540)));
				dataList[i + index].image.release();
				break;
			}
		}
	}
	catch (std::exception ex)
	{
		cout << " assertion failed or index error" << endl;
		writeErrorMessage(to_string(__LINE__) + "  assertion failed or index error : " + (string)ex.what());
	}
	catch (cv::Exception& e) //opencv 예외
	{
		writeErrorMessage(to_string(__LINE__) + " copyTo opencv error :" + (string)e.what());
	}


	return resultMat;

}

//스레드에서 동작
void  CMFCApplication1Dlg::detect()
{
	Detector detector("C:\\Program Files (x86)\\Twowinscom\\Parking Guidance\\Detector\\yyy.cfg", "C:\\Program Files (x86)\\Twowinscom\\Parking Guidance\\Detector\\yyy_4144.weights");
	//Detector detector("yyy.cfg", "yyy_4144.weights");

	vector<bbox_t> result_vec;

	CreateDirectory(_T(".\\downImage"), NULL);
	//시작

	try {
		while (true)
		{
			//초기화
			//readList.clear();
			dataList.clear();
			

			//time_t start_t = clock();
			getHttpPath();			//이미지 다운로드
			cout << "download complete" << endl;
			//start_t = clock() - start_t;
			//cout << "-------:send:--------- \n" << (float)(start_t) / CLOCKS_PER_SEC << "\n----------------------------\n" << endl;
			plateList.clear();


			//4개가 되면 합치고 판별
			//index += 4, count += 1, cars = 인식된 차 수
			int count = 0, index = 0, cars;
			while (index < dataList.size())
			{
				Mat result = concatImg(index);				//이미지 합침
				result_vec = detector.detect(result, 0.4);		//분석
				result.release();

				// 분석된 결과를 4개의 이미지 roi 좌표랑 분석
				// 이미지 불러오기 과정 없음, 좌표만 분석
				for (int i = 0; i < 4; i++)
				{

					count = index + i;
					if (count >= dataList.size())
						break;

					cars = 0;

					//1장에 최대 roi 3면씩존재 하므로 3번까지 반복
					for (int j = 0; j < 3; j++)
					{	
						int roi_x = dataList[count].x[j] / 2;		//분석된 좌표는 resize된좌표, roi 좌표는 원래 좌표이므로 좌표 비율 맞춤
						int roi_y = dataList[count].y[j] / 2;		//가로 세로 1/2 씩 전체 크기 1/4 

						switch (i)										//이미지 합쳐질시 왼쪽위, 오른쪽위, 오른쪽아래등 좌표 + 변화가 생기는거 맞춤
						{
						case 0:break;									//왼쪽위
						case 1:roi_x += 960; break;						//오른쪽위
						case 2:roi_y += 540; break;						//왼쪽아래
						case 3:roi_x += 960; roi_y += 540; break;		//오른쪽아래
						}

						//판별된 차 사각형 좌표중 하나씩
						for (bbox_t box : result_vec)
						{
							if (box.obj_id == 0)			// 0 = car, 1 = plate
							{
								if (box.w < 650 && box.h < 450 && box.w > 150 && box.h > 120)	// 객체가 너무 작거나 큰경우 차가 아니므로 필터링
								{
									int left = box.x;
									int top = box.y;
									int right = box.x + box.w;
									int bottom = box.y + box.h;
									int centerX = box.x + box.w / 2;
									int centerY = box.y + box.h / 2;

									//조건 1, 2 로 roi 내에 차량 유무 판단 정확도 보정 
									//조건1: 분석된 사각형내에 roi point 위치,  조건2: roi point 와 분석된사각형 center 가 너무 멀리 떨어져있으면 안됨

									if (roi_x != 0 && roi_y != 0 && roi_x > left + 20 && roi_x < right - 20 && roi_y > top + 20 && roi_y < bottom - 20)  //조건1  20은 margin 값
									{
										int roi_left = roi_x - 90, roi_right = roi_x + 90, roi_top = roi_y - 70, roi_bottom = roi_y + 70; // 

										if (centerX > roi_left && centerX < roi_right && centerY > roi_top && centerY < roi_bottom) //조건2
										{
											cars++;					//차량 roi point 에 존재
											break;
										}
									}
								}
							}
						}
					}

					dataList[count].cars = cars;					//리스트에 차량 개수 저장
					if (cars == dataList[count].areas)			//차 개수 =  roi 위치 개수 -> Red
						dataList[count].isRed = true;

					///* 번호판 영역 시작 삭제 *///

				}
				

				index += 4;
			}


			send();				//UI 에 보냄

		}
	}
	catch (exception e)
	{
		writeErrorMessage(to_string(__LINE__) + "  failed detect image " + (string)e.what());
	}

}



bool cmp(const bbox_t& b1, const bbox_t& b2)
{
	if (b1.x < b2.x)
		return true;
	else
		return false;
}

//차 인식 결과 UI 에 보냄
void send()
{
	try
	{
		for (int i = 0; i < dataList.size(); i++)
		{
			string sData;
			if(dataList[i].isDown)
				sData = "car_" + dataList.at(i).panicAddr + "_" + dataList.at(i).ip + "_" + (dataList.at(i).isRed ? "Red" : "Green") + "_" + to_string(dataList.at(i).cars) + "_" + to_string(dataList.at(i).areas);
			

			int size = sData.length();

			char temp[100] = {0,};
			strcpy(temp, sData.c_str());

			DWORD dwBytesToWrite = (DWORD)100;
			DWORD cbWritten;

			cout << sData << endl;

			waitKey(1);

			bool bWrite = false;
			bWrite = WriteFile(hPipe1, temp, dwBytesToWrite, &cbWritten, NULL);

			while (bWrite)	// 다 보내기를 기다림
				break;
		}
	}
	catch (exception ex) //일반 예외
	{
		writeErrorMessage(to_string(__LINE__) + " Send UI error :" + (string)ex.what());
	}
	catch (cv::Exception& e) //opencv 예외
	{
		writeErrorMessage(to_string(__LINE__) + " Send UI opencv error :" + (string)e.what());
	}

	//UI 만 비정상 종료시 같이 꺼지도록함 
	/*HWND isClose = ::FindWindow(NULL, _T("PanicManagementSystem"));
	if (isClose == NULL)
	{
		string time = getTime();
		string filename = time + "ExceptionLog.txt";
		ofstream outFile(filename, ios::app);

		outFile << time << " - " << "6. UI program is not running " << endl;
		outFile.close();

		exit(0);
	}*/
}


//차 인식 테스트용
void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec)
{
	//int const colors[1][3] = { { 1,0,1 } };

	for (auto &i : result_vec) {
		//cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::Scalar color(240, 240, 0, 100);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
	}
}
//번호 인식 테스트용
void draw_boxes2(cv::Mat mat_img, std::vector<bbox_t> result_vec)
{
	//int const colors[1][3] = { { 1,0,1 } };

	for (auto &i : result_vec) {
		//cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::Scalar color = cv::Scalar(0, 0, 255, 100);
		switch (i.obj_id)
		{
		case 0:  putText(mat_img, "0", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 1:  putText(mat_img, "1", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 2:  putText(mat_img, "2", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 3:  putText(mat_img, "3", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 4:  putText(mat_img, "4", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 5:  putText(mat_img, "5", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 6:  putText(mat_img, "6", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 7:  putText(mat_img, "7", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 8:  putText(mat_img, "8", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		case 9:  putText(mat_img, "9", Point(i.x, i.y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, color, 2, 1); break;
		}
	}
}


string getTime()
{
	time_t now = time(0);
	struct tm tstruct;
	char buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%y-%m-%d.%X:%M", &tstruct);
	return buf;
}
void writeErrorMessage(string msg)
{
	vector<string> time = split2(getTime(), '.');
	string filename = time.at(0) + " ExceptionLog.txt";
	ofstream outFile(filename, ios::app);

	outFile << time.at(0) <<"." << time.at(1) << " Line:" << msg << endl;
	outFile.close();
}
void CMFCApplication1Dlg::OnBnClickedOk()
{
	//test();
	//cout << "123123123123123123" << endl;
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	//getHttpPath();           cout << "다운 완료 " << endl;
	//detect();                cout << "판별 완료 " << endl;
	//send();                  cout << "전송 완료 " << endl;
}







// hide dialog 
void CMFCApplication1Dlg::OnWindowPosChanging(WINDOWPOS* lpwndpos)
{
	//lpwndpos->flags &= ~SWP_SHOWWINDOW;

	CDialogEx::OnWindowPosChanging(lpwndpos);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
}
