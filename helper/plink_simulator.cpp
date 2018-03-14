#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <iomanip>
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/random.hpp"
#include "boost/numeric/ublas/lu.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/triangular.hpp"
#include "boost/accumulators/accumulators.hpp"
#include "boost/accumulators/statistics/stats.hpp"
#include "boost/accumulators/statistics/mean.hpp"
#include "boost/accumulators/statistics/moment.hpp"
#include "boost/accumulators/statistics/variance.hpp"
namespace ublas = boost::numeric::ublas; 
using namespace std; 
using namespace boost::accumulators; 
#include <bits/stdc++.h>


bool InvertMatrix (const ublas::matrix<double>& input, ublas::matrix<double>& inverse) {
 	using namespace boost::numeric::ublas;
 	typedef permutation_matrix<std::size_t> pmatrix;
 	// create a working copy of the input
   	matrix<double> A(input);
 	// create a permutation matrix for the LU-factorization
 	pmatrix pm(A.size1());
 	// perform LU-factorization
 	int res = lu_factorize(A,pm);
 	if( res != 0 ) return false;
 	// create identity matrix of "inverse"
 	inverse.assign(ublas::identity_matrix<double>(A.size1()));
 	// backsubstitute to get the inverse
 	lu_substitute(A, pm, inverse);
 	return true;
}

double MatrixTrace (const ublas::matrix<double>& A, int n)
{

	double result =0; 
	for(int i=0; i<n; i++)
		result += A(i,i); 
	return result; 


}
int main(int argc, char const *argv[])
{

	if(argc<4){

		cout<<"Correct Usage ./simulator <Indv> <Snps> <h2g> <outFileHeader>  "<<endl; 
	}
	int n=atoi(argv[1]); 
	int m=atoi(argv[2]); 
	float h2g= atof(argv[3]); 
	 


	double panc[m]; //allele frequency
	boost::mt19937 gen; 
	boost::uniform_real<> dist(0.05,0.95); 
	boost::variate_generator<boost::mt19937&, boost::uniform_real<> > init(gen, dist); 
	for(int i=0; i<m; i++)
		panc[i]=init();  
	
	ublas::matrix<double> beta(m, 1); 
	ublas::matrix<double> env(n, 1); 

	boost::uniform_real<> dist2(0,1); 
	boost::variate_generator<boost::mt19937&, boost::uniform_real<> > geno(gen, dist2); 
//	geno.engine().seed(static_cast<unsigned int>(std::time(0))); 

	//generate effect size beta

	boost::normal_distribution<> dist3(0,sqrt(h2g/m));
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > effect_size(gen, dist3); 
	effect_size.engine().seed(static_cast<unsigned int>(std::time(0)));
	for(int i=0; i<m; i++)
		beta(i,0) =  effect_size(); 


//	cout<<"printing beta: "<<endl<<beta<<endl; 

	boost::normal_distribution<> dist4(0,sqrt(1-h2g)); 
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > e(gen, dist4);
	//e.engine().seed(static_cast<unsigned int>(std::time(0))); 
	for(int i=0; i<n; i++) 
		env(i,0)= e(); 

//	cout<<"printing env: "<<endl<<env<<endl; 
//	cout<<"printing pheno: "<<endl<<pheno<<endl;
	cout<<"writing to file..."<<endl;  
//	FILE *fpgeno; FILE *fppheno;
	FILE *fped, *fmap, *fppheno; 
	std::string plinkPrefix = argv[4];
	fped=fopen((plinkPrefix+".ped").c_str(), "w"); 
	fmap=fopen((plinkPrefix+".map").c_str(), "w");    
	fppheno = fopen((plinkPrefix+".pheno.plink").c_str(), "w"); 
	for(int i=0; i<m; i++)
	{
		fprintf(fmap,"%d %s", 1,"rs");
                fprintf(fmap, "%d %d %d \n", i,0, i);
	}
	fclose(fmap); 
	fprintf(fppheno, "%s %s %d \n", "FID", "IID", 1); 
	ublas::matrix<int> G(n, m); 
	for(int i=0; i<n; i++){
	//simulate each individual
		fprintf(fped, "%d %d %d %d %d %d", i,1,0,0,0,0); 
		ublas::matrix<int> G1(1,m); 
		ublas::matrix<int> G2(1,m); 
		//accumulator_set<double, stats<tag::mean, tag::moment<2> > > acc; 
		for(int j=0; j<m; j++)
		{
		double temp1=geno(); double temp2=geno(); 
			G1(0,j) = (temp1<=panc[j]); G2(0,j)=(temp2<=panc[j]); 
			int val= G1(0,j)+ G2(0,j);
		//	acc(val); 
			G(i,j) = val; 
			bool temp=(j < m/2); 
			if(val==0)
			{
				if(temp)
					fprintf(fped, " %c %c", 'A','A'); 
				else 	fprintf(fped, " %c %c", 'C', 'C'); 
			}
			else if(val==1) 
			{
				if(temp) 
					fprintf(fped, " %c %c", 'A', 'G'); 
				else 	fprintf(fped, " %c %c", 'C', 'T');
			}
			else
			{
				if(temp)
					fprintf(fped, " %c %c", 'G', 'G'); 
				else 	fprintf(fped, " %c %c", 'T', 'T'); 
			}

		}
		fprintf(fped, "\n"); 
	}
	ublas::matrix<double> a(1,n, 1); 
	ublas::matrix<double> g_mean = prod(a, G);
	g_mean = g_mean / n; 
	ublas::matrix<double> g_var(1, m); 
	for(int i=0; i<m; i++){
		double q = 0.5 * g_mean(0,i); 
		g_var(0,i) = sqrt(2*q*(1-q)); 
		if(g_var(0,i)==0)
			g_var(0,i)=1; 
	}
	for(int i=0; i<n; i++)
	{
		ublas::matrix<double> X(1, m); 
		for(int j=0; j<m; j++)
			X(0, j) = (G(i,j)- g_mean(0,j)) / g_var(0,j) ; 
		ublas::matrix<double> pheno = prod(X, beta); 
		fprintf(fppheno, "%d %d %f", i, 1,pheno(0,0)+env(i,0));
		fprintf(fppheno, "\n"); 
	}
	 
	fclose(fped); 
	fclose(fppheno); 
}


