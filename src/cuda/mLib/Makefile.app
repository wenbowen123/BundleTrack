#
# makefile setup
#


#
# Before this Makefile is included ...
#   $(NAME) should be module name
#   $(CCSRCS) should list C++ source files
#   $(CSRCS) should list C source files
#   $(MLIB_RELATIVE_PATH) should be the (relative) location of mLib
#   $(MLIB_EXTERNAL_RELATIVE_PATH) should be the (relative) location of the external libs
#
# For example ...
#   NAME=foo
#   CCSRCS=$(NAME).C \
#       foo1.C foo2.C foo3.C
#   CSRCS=foo4.c foo5.c
#   MLIB_RELATIVE_PATH=../mLib
#


#
# targets
#

OBJS=$(CCSRCS:.cpp=.o) $(CSRCS:.c=.o) 
INCS=$(HSRCS) $(CCSRCS:.cpp=.h) $(CSRCS:.c=.h)

#
# operating system, architecture type
#

OS=$(shell uname -s)
ARCH=$(shell uname -m)

#
# c flags
#

#CC=clang++
CC=g++
# clang below
#BASE_CFLAGS=-std=c++11 -stdlib=libc++ -U__STRICT_ANSI__ $(USER_CFLAGS) $(OS_CFLAGS) -Wall -I$(MLIB_RELATIVE_PATH)/include  -I$(MLIB_EXTERNAL_RELATIVE_PATH)/include 
BASE_CFLAGS=-fpermissive -std=c++11 $(USER_CFLAGS) $(OS_CFLAGS) -Wall -I$(MLIB_RELATIVE_PATH)/include  -I$(MLIB_EXTERNAL_RELATIVE_PATH)/include
DEBUG_CFLAGS=$(BASE_CFLAGS) -g
OPT_CFLAGS=$(BASE_CFLAGS) -O3 -DNDEBUG
CFLAGS=$(DEBUG_CFLAGS)

#
# directories
#

RELEASE_DIR=release
EXE_DIR=bin
#LIB_DIR=$(MLIB_RELATIVE_PATH)/external/lib64
LIB_DIR=$(MLIB_EXTERNAL_RELATIVE_PATH)/libsLinux

#
# default rules
#

.SUFFIXES: .cpp .C .c .o

.cpp.o:
	$(CC) $(CFLAGS) -c $<

.C.o:
	$(CC) $(CFLAGS) -c $<

.c.o:
	gcc $(CFLAGS) -c $<

#
# target app
#

EXE = $(EXE_DIR)/$(NAME)



#
# link options
#

#BASE_LDFLAGS=-lc++ $(USER_LDFLAGS) -L$(LIB_DIR)
BASE_LDFLAGS=$(USER_LDFLAGS) -L$(LIB_DIR)
DEBUG_LDFLAGS=$(BASE_LDFLAGS) -g
OPT_LDFLAGS=$(BASE_LDFLAGS) -O 
LDFLAGS=$(DEBUG_LDFLAGS)


#
# Set up libs
#

#ifeq ("$(findstring CYGWIN,$(OS))", "CYGWIN")
##OPENGL_LIBS=-lglut32 -lglu32 -lopengl32
#OPENGL_LIBS=-lfglut -lglu32 -lopengl32 -lwinmm -lgdi32
#else ifeq ("$(OS)","Darwin")
#OPENGL_LIBS=-framework GLUT -framework opengl
#else
##OPENGL_LIBS=-lglut -lGLU -lGL -lm
#OPENGL_LIBS=-lfglut -lGLU -lGL -lm
#endif
#LIBS=$(USER_LIBS) $(PKG_LIBS) $(OPENGL_LIBS)
LIBS=$(USER_LIBS) $(PKG_LIBS)



#
# Make targets
#

opt:
	    $(MAKE) $(EXE) "CFLAGS=$(OPT_CFLAGS)" "LDFLAGS=$(OPT_LDFLAGS)"

debug:
	    $(MAKE) $(EXE) "CFLAGS=$(DEBUG_CFLAGS)" "LDFLAGS=$(DEBUG_LDFLAGS)"

$(EXE):	    $(OBJS) $(LIBDIR)
	    mkdir -p $(EXE_DIR)
	    $(CC) -o $(EXE) $(LDFLAGS) $(USER_OBJS) $(OBJS) $(LIBS)

#release:
#	    mkdir -p $(RELEASE_DIR)/apps
#	    mkdir $(RELEASE_DIR)/apps/$(NAME)1
#	    cp *.[cCIh] $(RELEASE_DIR)/apps/$(NAME)1
#	    cp Makefile $(RELEASE_DIR)/apps/$(NAME)1
#	    cp $(NAME).vcxproj $(RELEASE_DIR)/apps/$(NAME)1
#	    rm -r -f $(RELEASE_DIR)/apps/$(NAME)
#	    mv $(RELEASE_DIR)/apps/$(NAME)1 $(RELEASE_DIR)/apps/$(NAME)

clean:
	    -  rm -f *~ *.o $(EXE)






