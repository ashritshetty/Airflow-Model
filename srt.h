// .......................................................................
// srt.h
//
#define WIDTH 704
#define HEIGHT 192
#define DEPTH 192
#define DIRECTIONS 27
#define NODES ((WIDTH+1)*(HEIGHT+1)*(DEPTH+1))
#define SLICENODES ((WIDTH+1)*(HEIGHT+1))
#define FLOWS (NODES*DIRECTIONS)


// Defaults.  Override with .config.
int FINAL_TIME = 20000;
int V_DUMP_START = 100;
int V_DUMP_INTERVAL = 100;

#define SLICE (DEPTH/2) 
#define LWS 128

#define FREE 0
#define INPUT 1
#define EXIT 2
#define BOUNDARY 3
#define FFLOW 4
#define SOLID 5
#define FLOOR 6
#define ROOF 7
#define FRONT 8
#define BACK 9

#define NOISE 0.00001
#define SMALL 0.000001
#define LARGE 10000000000.0

#define WT0 (8.0/27.0)
#define WT1 (2.0/27.0)
#define WT2 (1.0/54.0)
#define WT3 (1.0/216.0)

#define SR2 (1.414213562373095)
#define SR3 (1.732050807568877)

#define LEGAL(i,j,k) (i>=0 && i<=WIDTH && j>=0 && j<=HEIGHT && k>=0 && k<=DEPTH)

#define cstore(i,j,k) ((i)*((HEIGHT+1)*(DEPTH+1))+(j)*(DEPTH+1)+(k))

#define store(i,j,k,m) ((i)*((HEIGHT+1)*(DEPTH+1)*DIRECTIONS)+(j)*\
	((DEPTH+1)*DIRECTIONS)+(m)*(DEPTH+1)+(k))

struct rvector {
        double x, y, z;
        };

typedef struct {
	double del[DIRECTIONS];
	struct rvector nrml[DIRECTIONS];
	// This must be 8-byte aligned.
	int i,j,k,l;
	} rrnode;

struct ivector {
	int x, y, z;	
	}; 

struct tri {
	struct rvector v0, v1, v2;
	};

double host_link_length[DIRECTIONS] = {
	0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
	SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2,
	SR3, SR3, SR3, SR3, SR3, SR3, SR3, SR3
	};

__constant__ double link_length[DIRECTIONS] = {
	0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
	SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2, SR2,
	SR3, SR3, SR3, SR3, SR3, SR3, SR3, SR3
	};

double host_link_weight[DIRECTIONS] = {
	WT0, WT1, WT1, WT1, WT1, WT1, WT1, 
	WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2,
	WT3, WT3, WT3, WT3, WT3, WT3, WT3, WT3
	};

__constant__ double link_weight[DIRECTIONS] = {
	WT0, WT1, WT1, WT1, WT1, WT1, WT1, 
	WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2, WT2,
	WT3, WT3, WT3, WT3, WT3, WT3, WT3, WT3
	};

struct ivector host_ci[DIRECTIONS] = {
	 0, 0, 0,  1, 0, 0,  -1, 0, 0,  0, 1, 0,  0,-1, 0,  0, 0, 1,  0, 0,-1,
     	 1, 1, 0, -1, 1, 0,   1,-1, 0, -1,-1, 0,  1, 0, 1, -1, 0, 1,  1, 0,-1, 
	-1, 0,-1,  0, 1, 1,   0,-1, 1,  0, 1,-1,  0,-1,-1,  1, 1, 1, -1, 1, 1,
	 1,-1, 1, -1,-1, 1,   1, 1,-1, -1, 1,-1,  1,-1,-1, -1,-1,-1
	};

__constant__ struct ivector ci[DIRECTIONS] = {
	 0, 0, 0,  1, 0, 0,  -1, 0, 0,  0, 1, 0,  0,-1, 0,  0, 0, 1,  0, 0,-1,
     	 1, 1, 0, -1, 1, 0,   1,-1, 0, -1,-1, 0,  1, 0, 1, -1, 0, 1,  1, 0,-1, 
	-1, 0,-1,  0, 1, 1,   0,-1, 1,  0, 1,-1,  0,-1,-1,  1, 1, 1, -1, 1, 1,
	 1,-1, 1, -1,-1, 1,   1, 1,-1, -1, 1,-1,  1,-1,-1, -1,-1,-1
	};

__constant__ int opp[DIRECTIONS] = {
	0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
	26, 25, 24, 23, 22, 21, 20, 19
	};

#define omega (1.82802)

