mLib is a library to support research projects, and has been used in a large number of publications. 
You are free to use this code with proper attribution in non-commercial applications (Please see LICENSE.txt).
For the possibilities of commercial use, please contact the authors.


CONTACT (feel free to contact us):
niessner@cs.stanford.edu
mdfisher@cs.stanford.edu
adai@cs.stanford.edu


INSTALLATION:
Make to check out mLib and mLibExternal on the same directory level; ideally, in you work folder. A typical file structure looks this:
E:\Work\mLib
E:\Work\mLibExternal
E:\Work\<project name>

Most code was developed under VS2013, but the library is cross platform and most modules run under Windows, Linux, and Mac. Example projects can be found in mLib\test.
In order to add mLib to your project create mLibInclude.h and mLibInclude.cpp files to. These file need to be added to your to your VisualStudio project or Makefile.


Requirements:
- All external libraries are in mLibExternal (the idea is that everybody is using the same library versions)
- DirectX SDK June 2010 (needs to be separately installed for Windows rendering)


We are also looking for active participation in maintaining and extending mLib. However, please when you are changing the API be aware that you might break other research projects.


Example of mLibInclude.h:
	#include "mLibCore.h"
	#include "mLibD3D11.h"
	#include "mLibD3D11Font.h"
	#include "mLibDepthCamera.h"
	#include "mLibANN.h"
	#include "mLibEigen.h"
	#include "mLibLodePNG.h"
	#include "mLibZLib.h"
	#include "mlibCGAL.h"
	#include "mLibOpenMesh.h"
	#include "mLibFreeImage.h"	//this must be included after OpenMesh; otherwise there is a crash
	using namespace ml;


Example of mLibInclude.cpp:
	#include "mLibCore.cpp"
	#include "mLibD3D11.cpp"
	#include "mLibLodePNG.cpp"
	#include "mLibZLib.cpp"