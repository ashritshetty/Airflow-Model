
__global__ void bounce(float* to, float* from, rrnode *dev_bounce, 
	unsigned char *ncls, int revcount)
{
	int i, j, k, x, y, z;
	int n= get_global_id(0);
	if(n>=revcount)
		{
			//printf("Overflow n=%d",n);
			return; 
		}
	i=dev_bounce[n].i;
	j=dev_bounce[n].j;
	k=dev_bounce[n].k;
	
	
	for(int l=0;l<DIRECTIONS;++l){
		if(dev_bounce[n].del[l] > -1){
			x = i - ci[l].x;
			y = j - ci[l].y;
			z = k - ci[l].z; 
			if(ncls[cstore(x,y,z)]==FFLOW){
				to[store(x,y,z,opp[l])] = from[store(x,y,z,l)];
			}
		}	
	}
		
		
	
//printf("I am in bounce");
return;
}

