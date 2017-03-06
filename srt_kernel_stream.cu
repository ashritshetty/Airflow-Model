__global__ void stream(float* to, float* from, unsigned char *ncls)
{
	int i, j, k, x, y, z;
	i = get_global_id(0);
	j = get_global_id(1);
	k = get_global_id(2);

	if(ncls[cstore(i,j,k)] == FREE || ncls[cstore(i,j,k)] == FFLOW){
		for(int l=0;l<DIRECTIONS;++l){
			x = i - ci[l].x;
			y = j - ci[l].y;
			z = k - ci[l].z;
			if(LEGAL(x,y,z))
				to[store(i,j,k,l)] = from[store(x,y,z,l)];
		}
		
	}
//printf("I am in stream");
return;
}
