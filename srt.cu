// 
// srt.cu - a single relaxation time, LB solution to Navier-Stokes
// 
//
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <values.h>
#include <sys/time.h>
#include <signal.h>
#include <stdint.h>
#include <unistd.h>

#include "srt.h"
#include "srt_kernels.cu"

// CUDA global memory pointers:
float *f[2];
rrnode *dev_bounce;
unsigned char *dev_nclass;

// Large host arrays:
float *host_f, *vfield;
int *host_rindex;
rrnode *host_bounce = NULL;

// Class array:
unsigned char nclass[NODES];

// This holds the Maxwell-Boltzmann equilibrium for rho=1, u=(0.1,0,0).
double eqvnode[DIRECTIONS];

// This holds the Maxwell-Boltzmann equilibrium for rho=1, u=(0,0,0), which
// is just the link weights.
double eqznode[DIRECTIONS];

// Parameters
double max_target_dim = 40.0;
struct rvector override_scale = {0.0, 0.0, 102.4};
struct rvector target_xlate = {200.5, 112.5, -128.5};

double dot(struct rvector u, struct rvector v)
{
return(u.x*v.x + u.y*v.y + u.z*v.z);
}

double vlength(struct rvector u)
{
return(sqrt(dot(u,u)));
}

struct rvector cross(struct rvector a, struct rvector b)
{
struct rvector c;
c.x = a.y*b.z-a.z*b.y;
c.y = -a.x*b.z+a.z*b.x;
c.z = a.x*b.y-a.y*b.x;
return(c);
}

double host_dotci(int i,struct rvector u)
{
struct rvector ci;
ci.x = (double)(host_ci[i].x);
ci.y = (double)(host_ci[i].y);
ci.z = (double)(host_ci[i].z);
return(dot(ci,u));
}

double genrand()
{
return(((double)(random()+1))/2147483649.);
}

// This is not called until node classes are set by set_initial_nclass( ) and
// find_boundary_nodes( ).
void init_lattice()
{
int i,j,k,m,nc;

for(i=0;i<=WIDTH;i++){
	for(j=0;j<=HEIGHT;j++){
		for(k=0;k<=DEPTH;k++){
			nc = nclass[cstore(i,j,k)];
			if(nc==SOLID || nc==BOUNDARY){
				for(m=0;m<DIRECTIONS;m++){
					host_f[store(i,j,k,m)] = eqznode[m];
					}
				}
			else {
				for(m=0;m<DIRECTIONS;m++){
					host_f[store(i,j,k,m)] = eqvnode[m]
						+ NOISE*(2.0*genrand()-1.0);
					}
				}
			}
		}
	}
}

void set_initial_nclass()
{
int i,j,k;

for(i=0;i<=WIDTH;i++){
	for(j=0;j<=HEIGHT;j++){
		for(k=0;k<=DEPTH;k++){
			nclass[cstore(i,j,k)] = FREE;
			}
		}
	}

// Overwrite front and back.
for(i=0;i<=WIDTH;i++){
	for(j=0;j<=HEIGHT;j++){
		nclass[cstore(i,j,0)] = BACK;
		nclass[cstore(i,j,DEPTH)] = FRONT;
		}
	}

// Overwrite floor and roof.
for(i=0;i<=WIDTH;i++){
	for(k=0;k<=DEPTH;k++){
		nclass[cstore(i,0,k)] = FLOOR;
		nclass[cstore(i,HEIGHT,k)] = ROOF;
		}
	}

// Overwrite input and exit.
for(j=0;j<=HEIGHT;j++){
	for(k=0;k<=DEPTH;k++){
		nclass[cstore(0,j,k)] = INPUT;
		nclass[cstore(WIDTH,j,k)] = EXIT;
		}
	}
}

double get_equilibrium_f(double rho, struct rvector u, int m)
{
double ddot;
ddot = host_dotci(m,u);
return(host_link_weight[m]*rho*(1.0+3.0*ddot+4.5*ddot*ddot-1.5*dot(u,u)));
}

// Note that the .obj file must specify triangles.
struct tri *trilist;
struct rvector *vlist;

int hits(struct rvector sp, struct rvector dp, int idx, double *tptr,int *insp,
	struct rvector *pnrml)
{
struct rvector a,b,c,d,nrml;
double t, u, v, base, vsize;

a.x = dp.x - sp.x;
a.y = dp.y - sp.y;
a.z = dp.z - sp.z;
b.x = trilist[idx].v2.x - sp.x;
b.y = trilist[idx].v2.y - sp.y;
b.z = trilist[idx].v2.z - sp.z;
c.x = trilist[idx].v2.x - trilist[idx].v0.x;
c.y = trilist[idx].v2.y - trilist[idx].v0.y;
c.z = trilist[idx].v2.z - trilist[idx].v0.z;
d.x = trilist[idx].v2.x - trilist[idx].v1.x;
d.y = trilist[idx].v2.y - trilist[idx].v1.y;
d.z = trilist[idx].v2.z - trilist[idx].v1.z;

nrml = cross(c,d);
base = dot(a,nrml);
if(fabs(base)<SMALL) return(0);
if(base > 0.0) *insp=1;
else *insp=0;
t = dot(b,nrml)/base;
if(t<SMALL) return(0);
u = dot(a,cross(b,d))/base;
v = dot(a,cross(c,b))/base;
if((u<0.0) || (v<0.0) || ((u+v)>1.0)) return(0);
vsize = vlength(nrml);
nrml.x /= vsize;
nrml.y /= vsize;
nrml.z /= vsize;
*pnrml = nrml;
*tptr = t;
return(1);
}

int find_boundary_nodes(int fcount)
{
int i, j, k, m, itri, nc, slot, xslot, new_winner;
int tx, ty, tz;
struct rvector sp, dp, nrml;
double del; 
double minx, maxx, miny, maxy, minz, maxz;
int imin, imax, jmin, jmax, kmin, kmax;
int inside, revcount, rindex;

// Now determine, for each triangle, which links hit it, and mark those 
// links as reverse directions.  Note that node classification can change 
// because a given node may be included in the rasterization of more than 
// one triangle. The closest hit determines it.
//
// Since objects can push through walls, we also need to prevent
// rewriting the labels on wall nodes.  It hoses the flood fill.
//
// Since we can't afford the space to store normals for every grid node,
// we have to do the compression here. This means we have to do two passes,
// where the first counts how many slots we'll need.
fprintf(stderr,"finding object boundary\n");
fprintf(stderr,"PASS 1\n");
slot = 0;
for(itri=0;itri<fcount;itri++){
	if(itri%1000==0) fprintf(stderr,"%d/%d done\n",itri,fcount);
	maxx = minx = trilist[itri].v0.x;
	maxy = miny = trilist[itri].v0.y;
	maxz = minz = trilist[itri].v0.z;
	if(trilist[itri].v1.x > maxx) maxx = trilist[itri].v1.x;
	if(trilist[itri].v1.x < minx) minx = trilist[itri].v1.x;
	if(trilist[itri].v1.y > maxy) maxy = trilist[itri].v1.y;
	if(trilist[itri].v1.y < miny) miny = trilist[itri].v1.y;
	if(trilist[itri].v1.z > maxz) maxz = trilist[itri].v1.z;
	if(trilist[itri].v1.z < minz) minz = trilist[itri].v1.z;
	if(trilist[itri].v2.x > maxx) maxx = trilist[itri].v2.x;
	if(trilist[itri].v2.x < minx) minx = trilist[itri].v2.x;
	if(trilist[itri].v2.y > maxy) maxy = trilist[itri].v2.y;
	if(trilist[itri].v2.y < miny) miny = trilist[itri].v2.y;
	if(trilist[itri].v2.z > maxz) maxz = trilist[itri].v2.z;
	if(trilist[itri].v2.z < minz) minz = trilist[itri].v2.z;
	imax = (int)(maxx+SR3+1);
	jmax = (int)(maxy+SR3+1);
	kmax = (int)(maxz+SR3+1);
	imin = (int)(minx-SR3-1);
	jmin = (int)(miny-SR3-1);
	kmin = (int)(minz-SR3-1);

	for(i=imin;i<=imax;i++){
		for(j=jmin;j<=jmax;j++){
			for(k=kmin;k<=kmax;k++){
				if(!LEGAL(i,j,k)) continue;
				// Most classes are not set yet.
				// Here we want only class FREE.
				nc = nclass[cstore(i,j,k)];
				if(nc!=FREE) continue;
				sp.x = (double)(i); 
				sp.y = (double)(j); 
				sp.z = (double)(k);
				for(m=1;m<DIRECTIONS;m++){
					tx = i+host_ci[m].x;
					ty = j+host_ci[m].y;
					tz = k+host_ci[m].z;
					dp.x = (double)(tx);
					dp.y = (double)(ty);
					dp.z = (double)(tz);
					if(hits(sp,dp,itri,&del,&inside,&nrml)){
						if(del>1.0) continue; // Hit beyond link.
						if((rindex=host_rindex[cstore(i,j,k)])==-1){
							// New slot needed.
							host_rindex[cstore(i,j,k)] = slot++;
							break;
							}
						}
					}
				}
			}
		}
	}
revcount = slot;

host_bounce = (rrnode *)calloc(revcount,sizeof(rrnode));
// Set all links to non-broken.
for(slot=0;slot<revcount;slot++) {
	for(m=0;m<DIRECTIONS;m++){
		host_bounce[slot].del[m] = -1.0;
		}
	}
// Reset reverse index.
for(i=0;i<NODES;i++) host_rindex[i] = -1;

fprintf(stderr,"PASS 2\n");
slot = 0;
for(itri=0;itri<fcount;itri++){
	if(itri%1000==0) fprintf(stderr,"%d/%d done\n",itri,fcount);
	maxx = minx = trilist[itri].v0.x;
	maxy = miny = trilist[itri].v0.y;
	maxz = minz = trilist[itri].v0.z;
	if(trilist[itri].v1.x > maxx) maxx = trilist[itri].v1.x;
	if(trilist[itri].v1.x < minx) minx = trilist[itri].v1.x;
	if(trilist[itri].v1.y > maxy) maxy = trilist[itri].v1.y;
	if(trilist[itri].v1.y < miny) miny = trilist[itri].v1.y;
	if(trilist[itri].v1.z > maxz) maxz = trilist[itri].v1.z;
	if(trilist[itri].v1.z < minz) minz = trilist[itri].v1.z;
	if(trilist[itri].v2.x > maxx) maxx = trilist[itri].v2.x;
	if(trilist[itri].v2.x < minx) minx = trilist[itri].v2.x;
	if(trilist[itri].v2.y > maxy) maxy = trilist[itri].v2.y;
	if(trilist[itri].v2.y < miny) miny = trilist[itri].v2.y;
	if(trilist[itri].v2.z > maxz) maxz = trilist[itri].v2.z;
	if(trilist[itri].v2.z < minz) minz = trilist[itri].v2.z;
	imax = (int)(maxx+SR3+1);
	jmax = (int)(maxy+SR3+1);
	kmax = (int)(maxz+SR3+1);
	imin = (int)(minx-SR3-1);
	jmin = (int)(miny-SR3-1);
	kmin = (int)(minz-SR3-1);

	for(i=imin;i<=imax;i++){
		for(j=jmin;j<=jmax;j++){
			for(k=kmin;k<=kmax;k++){
				if(!LEGAL(i,j,k)) continue;
				nc = nclass[cstore(i,j,k)];
				if((nc!=FREE)&&(nc!=BOUNDARY)&&(nc!=FFLOW)) continue;
				sp.x = (double)(i); 
				sp.y = (double)(j); 
				sp.z = (double)(k);
				for(m=1;m<DIRECTIONS;m++){
					tx = i+host_ci[m].x;
					ty = j+host_ci[m].y;
					tz = k+host_ci[m].z;
					dp.x = (double)(tx);
					dp.y = (double)(ty);
					dp.z = (double)(tz);
					new_winner = 0;
					if(hits(sp,dp,itri,&del,&inside,&nrml)){ 
						if(del>1.0) continue;
						// 1 ray unit != 1 grid unit. Convert to grid distance.
						del *= host_link_length[m];
						if((rindex=host_rindex[cstore(i,j,k)])==-1){
							// We've not seen this node.
							host_bounce[slot].i = i;
							host_bounce[slot].j = j;
							host_bounce[slot].k = k;
							host_bounce[slot].del[m] = del;
							host_bounce[slot].nrml[m] = nrml;
							// Overall winner in direction slot 0.
							host_bounce[slot].del[0] = del;
							host_rindex[cstore(i,j,k)] = slot++;
							new_winner = 1;
							}
						else {
							// We have seen this node, but maybe not
							// this direction.
							if (host_bounce[rindex].del[m]<0.0 || 
								del<host_bounce[rindex].del[m]){
								host_bounce[rindex].del[m] = del;
								host_bounce[rindex].nrml[m] = nrml;
								if (del<host_bounce[rindex].del[0]){
									host_bounce[rindex].del[0] = del;
									new_winner = 1;
									}
								}
							}
						if(new_winner){
							// We have to reclassify (perhaps) on 
							// each hit.
							if(inside) nclass[cstore(i,j,k)] = BOUNDARY;
							else nclass[cstore(i,j,k)] = FFLOW;
							}
						}
					}
				}
			}
		}
	}
// Now go back and remove broken links from BOUNDARY or FFLOW 
// nodes that don't reach FFLOW nodes, since such links are not used
// in the bounce( ) kernel. 
for(slot=0;slot<revcount;slot++){
	i = host_bounce[slot].i;
	j = host_bounce[slot].j;
	k = host_bounce[slot].k;
	nc = nclass[cstore(i,j,k)];
	for(m=1;m<DIRECTIONS;m++){
		tx = i+host_ci[m].x;
		ty = j+host_ci[m].y;
		tz = k+host_ci[m].z;
		if((nc==BOUNDARY || nc==FFLOW) && 
			nclass[cstore(tx,ty,tz)]!=FFLOW){ 
			host_bounce[slot].del[m] = -1.0;
			}
		}
	}
// Do we have any broken links left?
for(slot=0;slot<revcount;slot++){
	minx = LARGE;
	for(m=1;m<DIRECTIONS;m++){
		if((del=host_bounce[slot].del[m])>0.0 && del<minx) minx=del;
		}
	if(minx<LARGE) host_bounce[slot].del[0] = minx; 
	else host_bounce[slot].del[0] = -1.0;
	}
// Finally, squeeze host_bounce[ ] list down.
slot=0; xslot=0;
while(xslot<revcount){
	if(host_bounce[xslot].del[0]>0.0){
		host_bounce[slot].i = host_bounce[xslot].i;
		host_bounce[slot].j = host_bounce[xslot].j;
		host_bounce[slot].k = host_bounce[xslot].k;
		for(m=0;m<DIRECTIONS;m++){
			host_bounce[slot].del[m] = host_bounce[xslot].del[m];
			host_bounce[slot].nrml[m] = host_bounce[xslot].nrml[m];
			}
		slot++;
		}
	xslot++;
	}
revcount = slot;
return(revcount);
}

void host_get_u(float *hf,int i,int j,int k,double *rptr, struct rvector *uptr)
{
int m;
double rho = 0.0;
struct rvector mo, u;

for(m=0;m<DIRECTIONS;m++) rho += (double)(hf[store(i,j,k,m)]);
mo.x = mo.y = mo.z = 0.0;
for(m=0;m<DIRECTIONS;m++){
        mo.x += host_ci[m].x*(double)(hf[store(i,j,k,m)]);
        mo.y += host_ci[m].y*(double)(hf[store(i,j,k,m)]);
        mo.z += host_ci[m].z*(double)(hf[store(i,j,k,m)]);
        }
u.x = mo.x/rho;
u.y = mo.y/rho;
u.z = mo.z/rho;
*rptr = rho;
*uptr = u;
}

float ReverseEndian(float val)
{
uint32_t ivalue = htobe32(*(uint32_t *)(&val));
return(*(float *)&ivalue);
}

void write_vtk_header(FILE* fptr, char* buf)
{
fprintf(fptr,"# vtk DataFile Version 2.0\n");
fprintf(fptr,"%s\n", buf);
fprintf(fptr,"BINARY\n");
fprintf(fptr,"DATASET STRUCTURED_POINTS\n");
// fprintf(fptr,"DIMENSIONS %d %d %d\n", WIDTH+1, HEIGHT+1, DEPTH+1);
fprintf(fptr,"DIMENSIONS %d %d %d\n", WIDTH+1, HEIGHT+1, 1);
fprintf(fptr,"SPACING 1 1 1\n");
fprintf(fptr,"ORIGIN 0 0 0\n");
fprintf(fptr,"POINT_DATA %d\n", SLICENODES);
fprintf(fptr,"VECTORS vfield float\n");
}

void save_velocity_field(int iteration)
{
int i,j;
FILE *fptr;
char buf[256];
double rho;
struct rvector u;
float *vptr;

cudaMemcpy(&host_f[0],&f[iteration%2][0],FLOWS*sizeof(float),cudaMemcpyDefault);
cudaDeviceSynchronize();

sprintf(buf,"%s/vfield.%d.vtk","vdir",iteration);

fptr=fopen(buf,"w");
write_vtk_header(fptr,buf);

vptr = vfield;
for(j=0;j<=HEIGHT;j++){
	for(i=0;i<=WIDTH;i++){
		host_get_u(host_f,i,j,SLICE,&rho,&u);
		*vptr++ = ReverseEndian((float)(u.x));
		*vptr++ = ReverseEndian((float)(u.y));
		*vptr++ = ReverseEndian((float)(u.z));
		}
	}

fwrite(vfield,SLICENODES,3*sizeof(float),fptr);
fclose(fptr);
}

void cleanup(int signum)
{
cudaFree(f[0]);
cudaFree(f[1]);
cudaFree(dev_bounce);
cudaFree(dev_nclass); 
cudaDeviceReset();
exit(0);
}

void go(int rvcount)
{
int t, from, to, pad;
dim3 stream_lws(1,4,32);
dim3 stream_ws(WIDTH,HEIGHT/4,DEPTH/32);
dim3 cascade_lws(1,4,32);
dim3 cascade_ws(WIDTH,HEIGHT/4,DEPTH/32);

pad = (rvcount/LWS+1);

to = 0;
from = 1;
for(t=1;t<=FINAL_TIME;t++){
	cudaDeviceSynchronize();
	cascade<<<cascade_ws,cascade_lws>>>(f[to],f[from],dev_nclass);
	cudaMemcpy(f[from],f[to],FLOWS*sizeof(float),cudaMemcpyDefault); 
	// This one has a 1D grid:
	bounce<<<pad,LWS>>>(f[to],f[from],dev_bounce,dev_nclass,rvcount);
	stream<<<stream_ws,stream_lws>>>(f[from],f[to],dev_nclass);  

        if(t>=V_DUMP_START && (t%V_DUMP_INTERVAL)==0){
                save_velocity_field(t);
                }
	}
}

void set_eqvalues(double rho, struct rvector u)
{
int m;
for(m=0;m<DIRECTIONS;m++) {
	eqvnode[m] = get_equilibrium_f(rho,u,m);
	eqznode[m] = host_link_weight[m];
	}
}

void buffers(int revcount)
{
long long bytes = 0;
cudaError_t err;

err = cudaMalloc(&f[0],FLOWS*sizeof(float));
if(!(err==cudaSuccess)) fprintf(stderr,"cudaMalloc f[0] failed\n");
cudaMemcpy(f[0],&host_f[0],FLOWS*sizeof(float),cudaMemcpyDefault);

err = cudaMalloc(&f[1],FLOWS*sizeof(float));
if(!(err==cudaSuccess)) fprintf(stderr,"cudaMalloc f[1] failed\n");
cudaMemcpy(f[1],&host_f[0],FLOWS*sizeof(float), cudaMemcpyDefault);

bytes += 2*((long long)(FLOWS*sizeof(float)));

err = cudaMalloc(&dev_bounce,revcount*sizeof(rrnode));
if(!(err==cudaSuccess)) fprintf(stderr,"cudaMalloc dev_bounce failed\n");
cudaMemcpy(dev_bounce,host_bounce,revcount*sizeof(rrnode),cudaMemcpyDefault);

bytes += (revcount)*sizeof(rrnode);

err = cudaMalloc(&dev_nclass,NODES*sizeof(unsigned char));
if(!(err==cudaSuccess)) fprintf(stderr,"cudaMalloc dev_nclass failed\n");
cudaMemcpy(dev_nclass,nclass,NODES*sizeof(unsigned char),cudaMemcpyDefault);

bytes += NODES*sizeof(unsigned char);

fprintf(stderr,"total allocated card memory: %lld\n",bytes);
}

void host_arrays()
{
int i;

host_f = (float *)calloc(FLOWS,sizeof(float));
if(host_f == NULL) fprintf(stderr,"oops\n");

host_rindex = (int *)calloc(NODES,sizeof(int));
if(host_rindex == NULL) fprintf(stderr,"oops\n");
for(i=0;i<NODES;i++) host_rindex[i] = -1;

vfield = (float *)calloc(NODES*3,sizeof(float));
if(vfield == NULL) fprintf(stderr,"oops\n");
}

// This loads an obj file, computes a bounding box, and places the bounding box 
// within the grid using a scale and a translate.  The .obj file is of the
// stripped-down variety, i.e., no material library,
// no normal indices, no texture indices, just vertices and faces.

int parse_geometry(char *filename)
{
char buf[512];
int vcount, fcount, iv0, iv1, iv2;
double minx, maxx, miny, maxy, minz, maxz;
double mdatadim, mtargetdim, x, y, z;
double xscale, yscale, zscale;

FILE *fptr;
fptr = fopen(filename,"r");
vcount = fcount = 0;
minx = miny = minz = 1000000000.0;
maxx = maxy = maxz = -1000000000.0;
while(fgets(buf,512,fptr)>0){
        if(buf[0]=='v'){
		vcount++;
		// Caution: engineers think z (last coordinate) is up.
		sscanf(buf,"v %lf %lf %lf",&x,&y,&z);
       		if(x<minx) minx = x;
                if(y<miny) miny = y;
                if(z<minz) minz = z;
                if(x>maxx) maxx = x;
                if(y>maxy) maxy = y;
                if(z>maxz) maxz = z;
		}
	else {
		if(buf[0]=='f') fcount++;
		}
	}
fprintf(stderr,"min %f %f %f\n",minx,miny,minz);
fprintf(stderr,"max %f %f %f\n",maxx,maxy,maxz);
// Enforce non-zero dimensions.
if(minx>=maxx) maxx += SMALL;
if(miny>=maxy) maxy += SMALL;
if(minz>=maxz) maxz += SMALL;

// Now scale and translate so that the data fits into the grid.
xscale = yscale = zscale = -1.0;
if(override_scale.x==0.0) xscale = maxx-minx;
if(override_scale.y==0.0) yscale = maxy-miny;
if(override_scale.z==0.0) zscale = maxz-minz;
mdatadim = xscale;
if(yscale>mdatadim) mdatadim = yscale;
if(zscale>mdatadim) mdatadim = zscale;

mtargetdim = max_target_dim;
xscale = yscale = zscale = mtargetdim/mdatadim;
if(override_scale.x>0.0) xscale = override_scale.x;
if(override_scale.y>0.0) yscale = override_scale.y;
if(override_scale.z>0.0) zscale = override_scale.z;
fprintf(stderr,"scales are %f %f %f\n",xscale,yscale,zscale);

vlist = (struct rvector *)calloc(vcount+1,sizeof(struct rvector));
trilist = (struct tri *)calloc(fcount,sizeof(struct tri));

rewind(fptr);
vcount = 0;
fcount = 0;
// This assumes all vertices appear before all faces.
while(fgets(buf,512,fptr)>0){
        if(buf[0]=='v'){
		vcount++;
		sscanf(buf,"v %lf %lf %lf",&x,&y,&z);
		vlist[vcount].x = xscale*(x - minx) + target_xlate.x;
		vlist[vcount].y = yscale*(y - miny) + target_xlate.y;
		vlist[vcount].z = zscale*(z - minz) + target_xlate.z;
		}
	else {
		if(buf[0]=='f'){ 
			sscanf(buf,"f %d %d %d",&iv0,&iv1,&iv2);
			trilist[fcount].v0 = vlist[iv0];
			trilist[fcount].v1 = vlist[iv1];
			trilist[fcount].v2 = vlist[iv2];
			fcount++;
			}
		}
	}
fclose(fptr);
return(fcount);
}

int qtop;
struct ivector *ffq;
#define FFQSIZE 10000000

void qpush(int i, int j, int k)
{
struct ivector val;
val.x = i;
val.y = j;
val.z = k;
if(qtop==FFQSIZE-1){
	fprintf(stderr,"queue overflow\n");
	exit(1);
	}
ffq[++qtop] = val;
}

struct ivector qpop()
{
return(ffq[qtop--]);
}

void qprocess()
{
int nc,i,j,k;
struct ivector val;

while(qtop>0){
	val = qpop();
	i = val.x;
	j = val.y;
	k = val.z;
	if(!LEGAL(i,j,k)) {
		fprintf(stderr,"eek %d %d %d\n",i,j,k);
		exit(1);
		}
	nc = nclass[cstore(i,j,k)];
	if(nc!=FREE) continue;
	nclass[cstore(i,j,k)] = SOLID;
	qpush(i+1,j,k);
	qpush(i,j+1,k);
	qpush(i,j,k+1);
	qpush(i-1,j,k);
	qpush(i,j-1,k);
	qpush(i,j,k-1);
	}
}

void flood_fill()
{
// Mark all nodes inside BOUNDARY nodes as SOLID nodes.
// We need to find one to start.
int i,j,k;
qtop = -1;
ffq = (struct ivector *)calloc(FFQSIZE,sizeof(struct ivector));

for(i=0;i<WIDTH;i++){
	for(j=0;j<HEIGHT;j++){
		for(k=0;k<DEPTH;k++){
			if(nclass[cstore(i,j,k)]==BOUNDARY){
				qpush(i+1,j,k);
				qpush(i,j+1,k);
				qpush(i,j,k+1);
				qpush(i-1,j,k);
				qpush(i,j-1,k);
				qpush(i,j,k-1);
				qprocess();
				}
			}
		}
	}
free(ffq);
return;
}

int main(int argc, char **argv)
{
struct rvector u = {0.1,0.0,0.0};
double rho = 1.0;
int revcount, tricount; 

srandom(123456789);
signal(SIGUSR1,cleanup);
host_arrays();
set_eqvalues(rho,u);
set_initial_nclass();

tricount = parse_geometry(argv[1]);
fprintf(stderr,"tricount is %d\n",tricount);
revcount = find_boundary_nodes(tricount);

flood_fill();
init_lattice();
fprintf(stderr,"allocating buffers\n");
buffers(revcount);
go(revcount);
cleanup(SIGUSR1);
return(0);
}


