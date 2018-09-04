/** 
 Mailman algorithm is written by Aman Agrawal 
 (Indian Institute of Technology, Delhi)
 RHE_reg is written by Yue Ariel Wu
 (UCLA)
*/
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector> 
//#include <random>

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include "time.h"

#include "genotype.h"
#include "mailman.h"
#include "arguments.h"
#include "helper.h"
#include "storage.h"

#include "boost/random.hpp"
#include "boost/accumulators/statistics/stats.hpp"
#include "boost/math/distributions/chi_squared.hpp"
#include "boost/math/distributions/normal.hpp"
#include "boost/math/special_functions.hpp"

#if SSE_SUPPORT==1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif

using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;
//Intermediate Variables
int blocksize;
double *partialsums;
double *sum_op;		
double *yint_e;
double *yint_m;
double **y_e;
double **y_m;


struct timespec t0;

//clock_t total_begin = clock();
MatrixXdr pheno;
MatrixXdr pheno_prime; 
MatrixXdr covariate;  
genotype g;
MatrixXdr geno_matrix; //(p,n)
int MAX_ITER;
int k,p,n;
int k_orig;

MatrixXdr c; //(p,k)
MatrixXdr x; //(k,n)
MatrixXdr v; //(p,k)
MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)
MatrixXdr sum2;
MatrixXdr sum;  


//solve Ax= b, use for b
//
MatrixXdr yy; 
MatrixXdr yKy; 
MatrixXdr Xy; 
//use for covariate
MatrixXdr WW; 

//use for missing phenotype
MatrixXdr pheno_mask2; 
vector<int> pheno_mask; 


options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize=true;
int accelerated_em=0;
double convergence_limit;
bool memory_efficient = false;
bool missing=false;
bool fast_mode = true;
bool text_version = false;
bool use_cov=false; 
bool reg = true;
bool gwas=false;
bool bpheno=false;   

vector<string> pheno_name; 

std::istream& newline(std::istream& in)
{
    if ((in >> std::ws).peek() != std::char_traits<char>::to_int_type('\n')) {
        in.setstate(std::ios_base::failbit);
    }
    return in.ignore();
}
int read_cov(bool std,int Nind, std::string filename, std::string covname, int exist_ind){
	ifstream ifs(filename.c_str(), ios::in); 
	std::string line; 
	std::istringstream in; 
	int covIndex = 0; 
	std::getline(ifs,line); 
	in.str(line); 
	string b;
	vector<vector<int> > missing; 
	int covNum=0;  
	while(in>>b)
	{
		if(b!="FID" && b !="IID"){
		missing.push_back(vector<int>()); //push an empty row  
		if(b==covname && covname!="")
			covIndex=covNum; 
		covNum++; 
		}
	}
	vector<double> cov_sum(covNum, 0); 
	if(covname=="")
	{
		covariate.resize(Nind, covNum); 
		cout<< "Read in "<<covNum << " Covariates.. "<<endl;
	}
	else 
	{
		covariate.resize(Nind, 1); 
		cout<< "Read in covariate "<<covname<<endl;  
	}

	
	int j=0; 
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line);
		string temp;
		in>>temp; in>>temp; //FID IID 
		for(int k=0; k<covNum; k++){
			
			in>>temp;
			if(temp=="NA")
			{
				if(pheno_mask[j]==1)
					missing[k].push_back(j);
				else covariate(j,k)=0; 
				continue; 
			} 
			double cur = atof(temp.c_str()); 
			if(cur==-9)
			{
				if(pheno_mask[j]==1)
					missing[k].push_back(j); 
				else covariate(j,k)=0; 
				continue; 
			}
			if(covname=="")
			{
				cov_sum[k]= cov_sum[k]+ cur*pheno_mask[j]; 
				covariate(j,k) = cur*pheno_mask[j]; 
			}
			else
				if(k==covIndex)
				{
					covariate(j, 0) = cur*pheno_mask[j];
					cov_sum[k] = cov_sum[k]+cur*pheno_mask[j]; 
				}
		}
		//if(j<10) 
		//	cout<<covariate.block(j,0,1, covNum)<<endl; 
		j++;
	}
	//compute cov mean and impute 
	for (int a=0; a<covNum ; a++)
	{
		int missing_num = missing[a].size(); 
		cov_sum[a] = cov_sum[a] / (exist_ind - missing_num);

		for(int b=0; b<missing_num; b++)
		{
                        int index = missing[a][b];
                        if(covname=="")
                                covariate(index, a) = cov_sum[a];
                        else if (a==covIndex)
                                covariate(index, 0) = cov_sum[a];
                } 
	}
	if(std)
	{
		MatrixXdr cov_std;
		cov_std.resize(1,covNum);  
		MatrixXdr sum = covariate.colwise().sum();
		MatrixXdr sum2 = (covariate.cwiseProduct(covariate)).colwise().sum();
		MatrixXdr temp;
//		temp.resize(Nind, 1); 
//		for(int i=0; i<Nind; i++)
//			temp(i,0)=1;  
		for(int b=0; b<covNum; b++)
		{
			cov_std(0,b) = sum2(0,b) + exist_ind*cov_sum[b]*cov_sum[b]- 2*cov_sum[b]*sum(0,b);
			cov_std(0,b) =sqrt((exist_ind-1)/cov_std(0,b)) ;
			double scalar=cov_std(0,b); 
			for(int j=0; j<Nind; j++)
			{
				if(pheno_mask[j]==1){
				covariate(j,b) = covariate(j,b)-cov_sum[b];  
				covariate(j,b) =covariate(j,b)*scalar;
				} 
			}
			//covariate.col(b) = covariate.col(b) -temp*cov_sum[b];
			
		}
	}	
	return covNum; 
}
int read_pheno2(int Nind, std::string filename,int pheno_idx){
//	pheno.resize(Nind,1); 
	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	std::istringstream in;  
	int phenocount=0; 
	vector<vector<int> > missing; 
//read header
	std::getline(ifs,line); 
	in.str(line); 
	string b; 
	while(in>>b)
	{
		if(b!="FID" && b !="IID"){
			phenocount++;
			missing.push_back(vector<int>());  
			pheno_name.push_back(b); 
		}
	}
	if(pheno_idx !=0)
		pheno_name[0] = pheno_name[pheno_idx-1];   
	vector<double> pheno_sum(phenocount,0); 
	if(pheno_idx !=0)
		pheno.resize(Nind,1); 
	else
		pheno.resize(Nind, phenocount);
	pheno_mask2.resize(Nind, 1); 
	int i=0;  
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line); 
		string temp;
		//fid,iid
		//todo: fid iid mapping; 
		in>>temp; in>>temp; 
		for(int j=0; j<phenocount;j++) {
			in>>temp;
			if(pheno_idx !=0 && j==(pheno_idx-1))
			{
				if(temp=="NA")
				{	
					pheno_mask.push_back(0); 
					pheno_mask2(i,0)=0; 
					pheno(i,0)=0; 
				}
				else
				{
					double cur = atof(temp.c_str()); 
					pheno(i,0)=cur; 
					pheno_mask.push_back(1); 
					pheno_mask2(i,0)=1; 
				}
			}
			if(pheno_idx ==0){
				if(temp=="NA")
				{
					pheno_mask.push_back(0); 
					pheno_mask2(i,0)=0; 
					missing[j].push_back(i); 
					continue; 
				} 
				double cur= atof(temp.c_str()); 
				pheno(i,j)=cur; 
				pheno_sum[j] = pheno_sum[j]+cur; 
			}
		}
		i++;
	}
	cout<<"pheno_mask: "<<endl;
        for(int i=0; i<10; i++)
               cout<<pheno_mask[i]<<endl;
	//cout<<"pheno_mask2:" <<endl<<pheno_mask2<<endl; 
	if(pheno_idx!=0)
		return 1; 
	//if use multiple phenotype together
	//fill missing with mean
	for(int a=0; a<phenocount; a++)
	{
		int missing_num= missing[a].size(); 
		double	pheno_avg = pheno_sum[a]/(Nind- missing_num); 
		//cout<<"pheno "<<a<<" avg: "<<pheno_avg<<endl; 
		for(int b=0 ; b<missing_num; b++)
		{
			int index = missing[a][b]; 
			pheno(index, a)= pheno_avg; 
		}
	}
	return phenocount; 
}


//solve for Ax=b efficiently, return A^{-1}b = x, where A is fixted to be sigma_g^2XX^T/M+sigma_eI 
void conjugate_gradient(int n, double vg, double ve,MatrixXdr &A,  MatrixXdr &b, MatrixXdr &x , int exist_ind){
	int k=0;
	double thres=0.00001;//1e-5 
	int max_iter=50; 
	MatrixXdr r0(n, 1);
	MatrixXdr r1(n, 1); 
	MatrixXdr p(n, 1); 
	MatrixXdr s(n, 1); 
	for(int i=0; i<n; i++)
	{	x(i,0)=0; 
		//p(i,0)=0; 
	}
	double temp=1; 
	double beta,alpha; 
	r0=b; 
	r1=b; 
	MatrixXdr mask_sum = A.colwise().sum(); 
	while(temp>thres && k<max_iter){
		k++; 
		if(k==1)
			p = b;
		else	
		{
			MatrixXdr temp1 = r0.transpose() * r0; 
			MatrixXdr temp2 = r1.transpose() * r1; 
			beta = temp2(0,0)/ temp1(0,0); 
			p = r1+ beta*p;	
		}
		//use mailman to compute s=Ap  = vg* XX^T p + ve * p  
		//		s = A*p ; 
		//G^Tp 
		MatrixXdr res(g.Nsnp, 1); 
//		multiply_y_pre(p, 1, res, false,exist_ind); 
		MatrixXdr res_sum = res.colwise().sum(); 
		//Sigma*G^Tp 
		res =res.cwiseProduct(stds);
		//X^Tp = Sigma*G^Tp - Sigma*Mp  
		MatrixXdr resid = means.cwiseProduct(stds); 
		resid = resid * res(0,0); 
		res = res - resid; 
		res = res.cwiseProduct(A); 
		//mask one chromsome
		res = res.cwiseProduct(stds); 
		//GX^Tp 
		MatrixXdr new_res = res.transpose(); 
		MatrixXdr XXp (1, g.Nindv); 
//		multiply_y_post(new_res, 1, XXp, false, exist_ind); 
		MatrixXdr inter = res.transpose()*means;
		//XX^Tp

		//double scalar = inter(0,0); 
		MatrixXdr temp_resid = MatrixXdr::Constant(g.Nindv, 1,1)*inter; 
		s = XXp.transpose() - temp_resid; 
		s = s/mask_sum(0,0); 
		s = s*vg; 
		s = s+ ve*p ;  
			


		MatrixXdr temp1 = r1.transpose() * r1; 
		MatrixXdr temp2 = p.transpose()*s; 
		alpha = temp1(0,0)/temp2(0,0); 
		x = x+ alpha*p; 
		MatrixXdr r2= r1; 
		r1 = r1 - alpha * s; 
		r0=r2; 
	//	cout<<r0(0,0)<<endl; 

		MatrixXdr z = r1.transpose()*r1; 
		temp  = z(0,0); 
	}

} 
void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means, int exist_int){
	for(int k_iter=0;k_iter<Ncol_op;k_iter++){
		sum_op[k_iter]=op.col(k_iter).sum();		
	}

			//cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on premultiply"<<endl;
			cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
			cout << "Segment size = " << g.segment_size_hori << endl;
			cout << "Matrix size = " <<g.segment_size_hori<<"\t" <<g.Nindv << endl;
			cout << "op = " <<  op.rows () << "\t" << op.cols () << endl;
		}
	#endif


	//TODO: Memory Effecient SSE FastMultipy

	for(int seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply(g.segment_size_hori,g.Nindv,Ncol_op,g.p[seg_iter],op,yint_m,partialsums,y_m);
		int p_base = seg_iter*g.segment_size_hori; 
		for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++ ){
			for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
				res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
		}
	}

	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply(last_seg_size,g.Nindv,Ncol_op,g.p[g.Nsegments_hori-1],op,yint_m,partialsums,y_m);		
	int p_base = (g.Nsegments_hori-1)*g.segment_size_hori;
	for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
			res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on premultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	for(int p_iter=0;p_iter<p;p_iter++){
 		for(int k_iter=0;k_iter<Ncol_op;k_iter++){		 
			res(p_iter,k_iter) = res(p_iter,k_iter) - (g.get_col_mean(p_iter)*sum_op[k_iter]);
			if(var_normalize)
				res(p_iter,k_iter) = res(p_iter,k_iter)/(g.get_col_std(p_iter,exist_int));		
 		}		
 	}	

}

void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res,bool subtract_means, int exist_ind){

	MatrixXdr op;
	op = op_orig.transpose();

	if(var_normalize && subtract_means){
		for(int p_iter=0;p_iter<p;p_iter++){
			for(int k_iter=0;k_iter<Nrows_op;k_iter++)		
				op(p_iter,k_iter) = op(p_iter,k_iter) / (g.get_col_std(p_iter, exist_ind));		
		}		
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on postmultiply"<<endl;
		}
	#endif
	
	int Ncol_op = Nrows_op;

	//cout << "ncol_op = " << Ncol_op << endl;

	int seg_iter;
	for(seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply_pre(g.segment_size_hori,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);
	}
	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply_pre(last_seg_size,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);

	for(int n_iter=0; n_iter<n; n_iter++)  {
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) {
			res(k_iter,n_iter) = y_e[n_iter][k_iter];
			y_e[n_iter][k_iter] = 0;
		}
	}
	
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on postmultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	double *sums_elements = new double[Ncol_op];
 	memset (sums_elements, 0, Nrows_op * sizeof(int));

 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		double sum_to_calc=0.0;		
 		for(int p_iter=0;p_iter<p;p_iter++)		
 			sum_to_calc += g.get_col_mean(p_iter)*op(p_iter,k_iter);		
 		sums_elements[k_iter] = sum_to_calc;		
 	}		
 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		for(int n_iter=0;n_iter<n;n_iter++)		
 			res(k_iter,n_iter) = res(k_iter,n_iter) - sums_elements[k_iter];		
 	}


}

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op ,MatrixXdr &res, int exist_ind){
	for(int p_iter=0;p_iter<p;p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++){
			double temp=0;
			for(int n_iter=0;n_iter<n;n_iter++)
				temp+= g.get_geno(p_iter,n_iter,var_normalize,exist_ind)*op(n_iter,k_iter);
			res(p_iter,k_iter)=temp;
		}
	}
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,int exist_ind){
	for(int n_iter=0;n_iter<n;n_iter++){
		for(int k_iter=0;k_iter<Nrows_op;k_iter++){
			double temp=0;
			for(int p_iter=0;p_iter<p;p_iter++)
				temp+= op(k_iter,p_iter)*(g.get_geno(p_iter,n_iter,var_normalize,exist_ind));
			res(k_iter,n_iter)=temp;
		}
	}
}

void multiply_y_pre_naive(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	res = geno_matrix * op;
}

void multiply_y_post_naive(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	res = op * geno_matrix;
}

void multiply_y_post(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,bool subtract_means,int exist_ind){
    if(fast_mode)
        multiply_y_post_fast(op,Nrows_op,res,subtract_means,exist_ind);
    else{
		if(memory_efficient)
			multiply_y_post_naive_mem(op,Nrows_op,res,exist_ind);
		else
			multiply_y_post_naive(op,Nrows_op,res);
	}
}

void multiply_y_pre(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means,int exist_ind){
    if(fast_mode)
        multiply_y_pre_fast(op,Ncol_op,res,subtract_means, exist_ind);
    else{
		if(memory_efficient)
			multiply_y_pre_naive_mem(op,Ncol_op,res,exist_ind);
		else
			multiply_y_pre_naive(op,Ncol_op,res);
	}
}
void compute_se(MatrixXdr &Xy, MatrixXdr &y,MatrixXdr &se, double h2g, double h2e,double tr_k2, int B , int exist_ind)
{
	//compute X^T y
	//input X^y[i] p*1 vector
	cout<<"p: "<<p << "  n: "<<n<<endl; 
	MatrixXdr zb=Xy;
	//compute XXy
	for(int j=0; j<p; j++)
		zb(j, 0)= zb(j,0)*stds(j,0); 
	MatrixXdr new_zb = zb.transpose(); 
	MatrixXdr new_res(1,n); 
	multiply_y_post_fast(new_zb, 1, new_res,false, exist_ind); 
	MatrixXdr new_resid(1,p); 
	MatrixXdr zb_scale_sum = new_zb*means; 
	new_resid= zb_scale_sum* MatrixXdr::Constant(1,n, 1); 
	MatrixXdr alpha = (new_res-new_resid).transpose() ;
	//compute Ky
	for(int j=0; j< n; j++)
		alpha(j,0)= alpha(j,0)/p ;
	//alpha = (K-I)y 
	alpha = alpha -y; 
	MatrixXdr res(p,1); 
	MatrixXdr resid(p,1); 
	MatrixXdr inter = means.cwiseProduct(stds); 
	multiply_y_pre_fast(alpha, 1, res, false, exist_ind); 
	for(int j=0; j<p;j++)
		res(j,0)=res(j,0)*stds(j,0); 
	inter = means.cwiseProduct(stds); 
	resid = inter * alpha.sum(); 
	MatrixXdr Xalpha(p,1); 
	Xalpha = res-resid;
	//Xy =res;
	double yKy = (Xalpha.array()*Xalpha.array()).sum() / p; 
	double temp =(alpha.array()*alpha.array()).sum();  
	double result = yKy*h2g+temp*h2e; 
	result = 2*result + h2g*h2g*tr_k2/10/B;
	//cout<<result; 
	result = sqrt(result) / (tr_k2-n);  
	MatrixXdr result1(1,1); 
	result1(0,0)=result;se=result1;  
}
void compute_b(bool use_cov, int pheno_num, MatrixXdr &y_sum ,int exist_ind){
	if(!use_cov){
	 if(pheno_num<10)
        {
                MatrixXdr res(g.Nsnp, pheno_num);
                multiply_y_pre(pheno,pheno_num,res,false, exist_ind);
                for(int i=0; i<pheno_num; i++){
                        MatrixXdr cur= res.block(0,i,g.Nsnp, 1);
                        res.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                }
                MatrixXdr resid(g.Nsnp, pheno_num);
                for(int i=0; i<pheno_num; i++)
                {
                        resid.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_sum(0,i);
                }
                Xy = res-resid;
                MatrixXdr temp = Xy.transpose() *Xy;
		yKy=temp.diagonal(); 
	}
	else
	{
		for(int i=0; i*10<pheno_num; i++){
                int col_num = 10;
                if( (pheno_num-i*10)<10)
                        col_num = pheno_num-i*10;
                MatrixXdr pheno_block = pheno.block( 0, i*10,g.Nindv, col_num);
                MatrixXdr res(g.Nsnp, col_num);
                multiply_y_pre(pheno_block,col_num,res,false, exist_ind);
                for(int j=0; j<col_num; j++){
                        MatrixXdr cur= res.block(0,j,g.Nsnp, 1);
                        res.block(0,j,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                }
                MatrixXdr resid(g.Nsnp, col_num);
                for(int j=0; j<col_num; j++)
                {
                        resid.block(0,j,g.Nsnp, 1) = means.cwiseProduct(stds)*y_sum(0,i*10+j);
                }
		Xy.block(0, i*10, g.Nsnp, col_num) = res-resid;
                MatrixXdr Xy_cur = Xy.block(0, i*10, g.Nsnp, col_num);
                MatrixXdr temp = Xy_cur.transpose() * Xy_cur;
                yKy.block(i*10, 0, col_num, 1)  = temp.diagonal();
	        }

	}
	yKy=yKy/g.Nsnp;
	} 
	if(use_cov)
        {
		MatrixXdr y_temp = pheno-covariate* WW * pheno_prime;
		for(int i=0; i<g.Nindv; i++)
			y_temp(i,0) = y_temp(i,0)*pheno_mask[i];
		cout<<"y_temp: "<<y_temp.block(0,0,10,1)<<endl;  
		//y_temp = y_temp.cwiseProcduct(pheno_mask); 
		MatrixXdr y_temp_sum=y_temp.colwise().sum();
                if(pheno_num<10)
                {
                        MatrixXdr res(g.Nsnp, pheno_num);
                        multiply_y_pre(y_temp,pheno_num,res,false,exist_ind);
                        for(int i=0; i<pheno_num; i++){
                                MatrixXdr cur= res.block(0,i,g.Nsnp, 1);
                                res.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                        }
                        MatrixXdr resid(g.Nsnp, pheno_num);
                          for(int i=0; i<pheno_num; i++)
                        {
                                resid.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_temp_sum(0,i);
                        }
                        Xy = res-resid;
                        MatrixXdr temp = Xy.transpose() *Xy;
                        yKy = temp.diagonal();


                }

		else{
                	for(int j=0; j*10<pheno_num; j++){
                        int col_num = 10;
                        if( (pheno_num-j*10)<10)
                                col_num = pheno_num-j*10;
                        MatrixXdr pheno_block = y_temp.block( 0, j*10,g.Nindv, col_num);
                        MatrixXdr res(g.Nsnp, col_num);

                        multiply_y_pre(pheno_block,col_num,res,false, exist_ind);
                        for(int i=0; i<col_num; i++){
                                MatrixXdr cur= res.block(0,i,g.Nsnp, 1);
                                res.block(0,i,g.Nsnp, 1)  = cur.cwiseProduct(stds);
                        }
                        MatrixXdr resid(g.Nsnp, col_num);
                        for(int i=0; i<col_num; i++)
                        {
                                resid.block(0,i,g.Nsnp, 1) = means.cwiseProduct(stds)*y_temp_sum(0,j*10+i);
                        }
                        Xy.block(0, j*10, g.Nsnp, col_num) = res-resid;
                        MatrixXdr Xy_cur = Xy.block(0, j*10, g.Nsnp, col_num);
                        MatrixXdr temp = Xy_cur.transpose()*Xy_cur;
                        yKy.block(j*10, 0, col_num, 1)  = temp.diagonal();


                	}
                }
                yKy = yKy/g.Nsnp;
        }

	yy= pheno.transpose() * pheno;
        if(use_cov)
                yy= yy- pheno_prime.transpose() * WW * pheno_prime;


}
pair<double,double> get_error_norm(MatrixXdr &c, int exist_ind){
	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(p,k);
	MatrixXdr q_t(k,p);
	q_t = Q.transpose();
	MatrixXdr b(k,n);
	multiply_y_post(q_t,k,b,true,exist_ind);
	JacobiSVD<MatrixXdr> b_svd(b, ComputeThinU | ComputeThinV);
	MatrixXdr u_l,d_l,v_l; 
	if(fast_mode)
        u_l = b_svd.matrixU();
    else
        u_l = Q * b_svd.matrixU();
	v_l = b_svd.matrixV();
	d_l = MatrixXdr::Zero(k,k);
	for(int kk=0;kk<k; kk++)
		d_l(kk,kk) = (b_svd.singularValues())(kk);
	
	MatrixXdr u_k,v_k,d_k;
	u_k = u_l.leftCols(k_orig);
	v_k = v_l.leftCols(k_orig);
	d_k = MatrixXdr::Zero(k_orig,k_orig);
	for(int kk =0 ; kk < k_orig ; kk++)
		d_k(kk,kk)  =(b_svd.singularValues())(kk);

	MatrixXdr b_l,b_k;
    b_l = u_l * d_l * (v_l.transpose());
    b_k = u_k * d_k * (v_k.transpose());

    if(fast_mode){
        double temp_k = b_k.cwiseProduct(b).sum();
        double temp_l = b_l.cwiseProduct(b).sum();
        double b_knorm = b_k.norm();
        double b_lnorm = b_l.norm();
        double norm_k = (b_knorm*b_knorm) - (2*temp_k);
        double norm_l = (b_lnorm*b_lnorm) - (2*temp_l);	
        return make_pair(norm_k,norm_l);
    }
    else{
        MatrixXdr e_l(p,n);
        MatrixXdr e_k(p,n);
        for(int p_iter=0;p_iter<p;p_iter++){
            for(int n_iter=0;n_iter<n;n_iter++){
                e_l(p_iter,n_iter) = g.get_geno(p_iter,n_iter,var_normalize, exist_ind) - b_l(p_iter,n_iter);
                e_k(p_iter,n_iter) = g.get_geno(p_iter,n_iter,var_normalize,exist_ind) - b_k(p_iter,n_iter);
            }
        }

        double ek_norm = e_k.norm();
        double el_norm = e_l.norm();
        return make_pair(ek_norm,el_norm);
    }
}



int main(int argc, char const *argv[]){

	//clock_t io_begin = clock();
    //clock_gettime (CLOCK_REALTIME, &t0);

	pair<double,double> prev_error = make_pair(0.0,0.0);
	double prevnll=0.0;

	parse_args(argc,argv);

	
	//TODO: Memory effecient Version of Mailman

	memory_efficient = command_line_opts.memory_efficient;
	text_version = command_line_opts.text_version;
	fast_mode = command_line_opts.fast_mode;
	missing = command_line_opts.missing;
	reg = command_line_opts.reg;
	//gwas=false; 
	gwas=command_line_opts.gwas; 
	cout<<"perform GWAS: "<<gwas<<endl; 
	bpheno = command_line_opts.bpheno; 
	//get number of individuals
	std::stringstream f2;
        f2 << command_line_opts.GENOTYPE_FILE_PATH << ".fam";
        g.read_fam (f2.str());
	//cout<<"ind: "<<g.Nindv<<endl; 	
	//get phenotype
	int pheno_idx = command_line_opts.pheno_idx;
	std::string filename=command_line_opts.PHENOTYPE_FILE_PATH;
        int pheno_num= read_pheno2(g.Nindv, filename, pheno_idx);
        int exist_ind =0;
        for(int i=0; i<g.Nindv; i++)
                exist_ind += pheno_mask[i];
	int cov_num=0 ;
        if(filename=="")
        {
                cout<<"No Phenotype File Specified"<<endl;
                return 0 ;
        }
        cout<< "Read in "<<pheno_num << " phenotypes"<<endl;
        if(pheno_idx!=0)
                cout<<"Using phenotype "<<pheno_name[pheno_idx-1]<<endl;
	cout<< "There are "<<exist_ind<< " individuals with no missing phenotypes"<<endl; 
	MatrixXdr VarComp(pheno_num,2);

	//if(gwas)
	//	fast_mode=false; //for now, gwas need the genotype matrix, and compute kinship constructed with one chrom leave out 
	if(!reg)
		fast_mode=false; //force save whole genome if non randomized  
	if(text_version){
		if(fast_mode)
			g.read_txt_mailman(command_line_opts.GENOTYPE_FILE_PATH,missing, exist_ind);
		else
			g.read_txt_naive(command_line_opts.GENOTYPE_FILE_PATH,missing,exist_ind);
	}
	else{
		g.read_plink(command_line_opts.GENOTYPE_FILE_PATH,missing,fast_mode,pheno_mask, exist_ind);
		
	}

	//TODO: Implement these codes.
	if(missing && !fast_mode){
		cout<<"Missing version works only with mailman i.e. fast mode\n EXITING..."<<endl;
		exit(-1);
	}
	if(fast_mode && memory_efficient){
		cout<<"Memory effecient version for mailman EM not yet implemented"<<endl;
		cout<<"Ignoring Memory effecient Flag"<<endl;
	}
	if(missing && var_normalize){
		cout<<"Missing version works only without variance normalization\n EXITING..."<<endl;
		exit(-1);
	}

    //MAX_ITER =  command_line_opts.max_iterations ; 
	int B = command_line_opts.batchNum; 
	k_orig = command_line_opts.num_of_evec ;
	debug = command_line_opts.debugmode ;
	float tr2= command_line_opts.tr2; 
	check_accuracy = command_line_opts.getaccuracy;
	var_normalize = true; 
	accelerated_em = command_line_opts.accelerated_em;
	k = k_orig + command_line_opts.l;
	k = (int)ceil(k/10.0)*10;
	command_line_opts.l = k - k_orig;
	p = g.Nsnp;
	n = g.Nindv;
	bool toStop=false;
		toStop=true;
	srand((unsigned int) time(0));
	c.resize(p,k);
	x.resize(k,n);
	v.resize(p,k);
	means.resize(p,1);
	stds.resize(p,1);
	sum2.resize(p,1); 
	sum.resize(p,1); 

//	geno_matrix.resize(p,n); 
//	g.generate_eigen_geno(geno_matrix, var_normalize); 

	if(!fast_mode ){
		geno_matrix.resize(p,n);
		g.generate_eigen_geno(geno_matrix,var_normalize,exist_ind);
		cout<<geno_matrix.data()<<endl; 
		cout<<geno_matrix.rows()<<endl; 
		cout<<geno_matrix.cols()<<endl; 
	}
	
		
	//clock_t io_end = clock();

	//TODO: Initialization of c with gaussian distribution
	c = MatrixXdr::Random(p,k);


	// Initial intermediate data structures
	blocksize = k;
	int hsegsize = g.segment_size_hori; 	// = log_3(n)
	int hsize = pow(3,hsegsize);		 
	int vsegsize = g.segment_size_ver; 		// = log_3(p)
	int vsize = pow(3,vsegsize);		 

	partialsums = new double [blocksize];
	sum_op = new double[blocksize];
	yint_e = new double [hsize*blocksize];
	yint_m = new double [hsize*blocksize];
	memset (yint_m, 0, hsize*blocksize * sizeof(double));
	memset (yint_e, 0, hsize*blocksize * sizeof(double));

	y_e  = new double*[g.Nindv];
	for (int i = 0 ; i < g.Nindv ; i++) {
		y_e[i] = new double[blocksize];
		memset (y_e[i], 0, blocksize * sizeof(double));
	}

	y_m = new double*[hsegsize];
	for (int i = 0 ; i < hsegsize ; i++)
		y_m[i] = new double[blocksize];

	for(int i=0;i<p;i++){
		means(i,0) = g.get_col_mean(i);
		stds(i,0) =1/ g.get_col_std(i,exist_ind);
		sum2(i,0) =g.get_col_sum2(i); 
		sum(i,0)= g.get_col_sum(i); 
	}
	
	cout<<"printing sum2: "<<endl<<sum2.block(0,0,10,1)<<endl; 
	cout<<"printing means: "<<endl<<means.block(0,0,10,1)<<endl; 
	cout<<"printing std: "<<endl<<stds.block(0,0,10,1)<<endl; 	
	ofstream c_file;
	if(debug){
		c_file.open((string(command_line_opts.OUTPUT_PATH)+string("cvals_orig.txt")).c_str());
		c_file<<c<<endl;
		c_file.close();
		printf("Read Matrix\n");
	}

	cout<<"Running on Dataset of "<<g.Nsnp<<" SNPs and "<<g.Nindv<<" Individuals"<<endl;

	#if SSE_SUPPORT==1
		if(fast_mode)
			cout<<"Using Optimized SSE FastMultiply"<<endl;
	#endif

	//read covariate
	
	std::string covfile=command_line_opts.COVARIATE_FILE_PATH;
        std::string covname=command_line_opts.COVARIATE_NAME;  
	if(covfile!=""){
		use_cov=true; 
		cov_num=read_cov(true,g.Nindv, covfile, covname,exist_ind); 
		cout<<"Covariate: "<<covariate.block(0,0,10,10)<<endl; 
	//	cout<<cov_num<<endl; 
	} 
	else if(covfile=="")
		cout<<"No Covariate File Specified"<<endl; 
	MatrixXdr y_sum=pheno.colwise().sum();
	double prevelance = y_sum(0,0) / exist_ind; 
	MatrixXdr y_mean = y_sum/exist_ind;
	for(int i=0; i<g.Nindv; i++) 
		pheno.block(i,0,1,pheno_num) =(pheno.block(i,0,1,pheno_num) - y_mean)*pheno_mask[i]; //center phenotype	
	y_sum=pheno.colwise().sum();
	//correctness check 
	//MatrixXdr zb = MatrixXdr::Random(1, g.Nsnp); 
	//MatrixXdr res(1,g.Nindv); 
	//multiply_y_post_fast(zb, 1, res, false); 
	//cout<< MatrixXdr::Constant(1,4,1)<<endl;  
	//compute y^TKy
	yKy.resize(pheno_num, 1); 
	Xy.resize(g.Nsnp, pheno_num);  
	if(use_cov)
	{
		//WW = inv( W^TW)
		cout<<"computing WW... "<<endl;
		WW = covariate.transpose()*covariate;
		cout<<"WW : "<<WW<<endl;  
		WW = WW.inverse(); 
		cout<<"WW inverse: "<<WW<<endl; 
		cout<<"finish  computing WW"<<endl; 
		//pheno_prime.resize(cov_num, pheno_num); 
		pheno_prime= covariate.transpose()* pheno; 
	}
//	MatrixXdr yKy(pheno_num, 1);
//	MatrixXdr Xy(g.Nsnp, pheno_num); 
	
	//compute yy, yKy
	if(reg) 
        	compute_b( use_cov, pheno_num, y_sum, exist_ind); 
	else
	{	
		yy = pheno.transpose() * pheno; 
		yKy = pheno.transpose()/g.Nsnp;
	//	cout<<"yky: "<<yKy<<endl;  
	}
	cout<<"yKy: "<<yKy<<endl; 
	cout<<"yy: "<<yy<<endl; 

	//compute tr[K]
	double tr_k =0 ;
	double tr_k_rsid =0; 
	//MatrixXdr temp = sum2 + g.Nindv* means.cwiseProduct(means) - 2 * means.cwiseProduct(sum);
	//temp = temp.cwiseProduct(stds);
	//temp = temp.cwiseProduct(stds); 
	//tr_k = temp.sum() / g.Nsnp;
	tr_k = exist_ind; 

//	cout<<"computing trace of K: "<<trK-it_begin<<endl;  
//	for (int j=0; j<g.Nsnp; j++)
//	{
//		double temp = sum2(j,0)+g.Nindv*means(j,0)*means(j,0)- 2*means(j,0)*sum(j,0); 
//		temp = temp * stds(j,0) * stds(j,0); 
//		tr_k += temp; 
//	} 	
//	tr_k = tr_k / g.Nsnp; 
	//compute tr[K^2]
	if(tr2<0){
	//compute/estimating tr_k2
	double tr_k2=0;
	//DiagonalMatrix<double,a> Sigma(a); 
	//Sigma.diagonal()=vec; 

	//clock_t it_begin =clock();
	if(reg){ 
	for(int i=0; i<B; i++){
		//G^T zb 
        	//clock_t random_step=clock(); 
		MatrixXdr zb= MatrixXdr::Random(g.Nindv, 10);
		zb = zb * sqrt(3); 
		for(int b=0; b<10; b++){
			MatrixXdr temp = zb.block(0,b,g.Nindv,1); 
			zb.block(0,b,g.Nindv, 1) = temp.cwiseProduct(pheno_mask2); 
		}
		MatrixXdr res(g.Nsnp, 10); 
		multiply_y_pre(zb,10,res, false, exist_ind);
		//sigma scale \Sigma G^T zb; compute zb column sum
		MatrixXdr zb_sum = zb.colwise().sum(); 
		//std::vector<double> zb_sum(10,0);  
		//res = Sigma*res;
		for(int j=0; j<g.Nsnp; j++)
		        for(int k=0; k<10;k++)
		             res(j,k) = res(j,k)*stds(j,0); 
	//	print_time();  
		//compute /Sigma^T M z_b
		MatrixXdr resid(g.Nsnp, 10);
		MatrixXdr inter = means.cwiseProduct(stds);
		resid = inter * zb_sum;
		MatrixXdr zb1(g.Nindv,10); 
		zb1 = res - resid; // X^Tzb =zb' 
	//	clock_t Xtzb = clock(); 
		//compute zb' %*% /Sigma 
		//zb = Sigma*zb ; 
		
              	for(int k=0; k<10; k++){
                  for(int j=0; j<g.Nsnp;j++){
                        zb1(j,k) =zb1(j,k) *stds(j,0);}}
                                              
		MatrixXdr new_zb = zb1.transpose(); 
		MatrixXdr new_res(10, g.Nindv);
		multiply_y_post(new_zb, 10, new_res, false, exist_ind); 
		//new_res =  zb'^T \Sigma G^T 10*N 
		MatrixXdr new_resid(10, g.Nindv); 
		MatrixXdr zb_scale_sum = new_zb * means;
		new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);  
		MatrixXdr Xzb = new_res- new_resid; 
		for( int b=0; b<10; b++)
		{
			MatrixXdr temp = Xzb.block(b,0,1, g.Nindv); 
			Xzb.block(b,0,1, g.Nindv) = temp.cwiseProduct(pheno_mask2.transpose()); 
		}
		if(use_cov)
		{
			MatrixXdr temp1 = WW * covariate.transpose() *Xzb.transpose(); 
			MatrixXdr temp = covariate * temp1;
			MatrixXdr Wzb  = zb.transpose() * temp;
			tr_k_rsid += Wzb.trace(); 
				
			Xzb = Xzb - temp.transpose(); 
		}
		tr_k2+= (Xzb.array() * Xzb.array()).sum();  
//		clock_t rest = clock(); 
	}
	tr_k2  = tr_k2 /10/g.Nsnp/g.Nsnp/B; 
	tr_k_rsid = tr_k_rsid/10/g.Nsnp/B; 
	}
	else{
	//	for(int i=0;i<n ;i++)
	//		for(int j=0; j<p; j++)
	//		{
	//			geno_matrix(j,i) = (geno_matrix(j,i)-means(j,0))*stds(j,0); 
	//		}
		cout<<"non reg"<<endl;
		MatrixXdr temp = geno_matrix.transpose()* geno_matrix; 
		temp = temp * temp ; 
		for(int i=0; i<n; i++)
			tr_k2 += temp(i,i); 
		tr_k2 = tr_k2/p/p; 
	}
	tr2=tr_k2; 
	}
	
	MatrixXdr A(2,2); 
	A(0,0)=tr2;
	A(0,1)=tr_k-tr_k_rsid; 
	A(1,0)= tr_k-tr_k_rsid; 
	A(1,1)=exist_ind-cov_num;
	cout<<A<<endl;   
	double vg,ve; 
	for(int i=0; i<pheno_num; i++){
		cout<< "Variance Component estimation for phenotype "<<i+1<<" "<<pheno_name[i]<<" :"<<endl; 
		MatrixXdr b(2,1); 
		b(0,0) = yKy(i,0); 
		b(1,0) = yy(i,i);
		cout<<"b: "<<endl << b <<endl;  
		MatrixXdr herit = A.colPivHouseholderQr().solve(b); 
		cout<<"V(G): "<<herit(0,0)<<endl;
		vg = herit(0,0); 
		ve = herit(1,0);
		VarComp(i,0)=herit(0,0); VarComp(i,1)=herit(1,0); 
		cout<<"V(e): "<<herit(1,0)<<endl; 
		cout<<"Vp "<<herit.sum()<<endl; 
		cout<<"V(G)/Vp: "<<herit(0,0)/herit.sum()<<endl;
		if(bpheno){
		cout<<"Prevelance: "<<prevelance<<endl; 
 		boost::math::normal m_normal(0.0, 1.0); 
		double t = quantile(m_normal,1-prevelance); 
		double c = pdf(m_normal, t); 
		c = c*c; 
		c= 1/c; 
		c = c* prevelance * (1-prevelance); 
		cout<<"Liability Scale: "<<herit(0,0)*c / herit.sum()<<endl; 
		}
	//	double c = g.Nindv* (tr2/g.Nindv - tr_k*tr_k/g.Nindv/g.Nindv); 
		
	//	cout<<"SE: "<<sqrt(2/c)<<endl; 
			

	//	if(reg){
	//	MatrixXdr se(1,1);
	//	MatrixXdr pheno_i = pheno.block(0,i, g.Nindv, 1); 
	//	if(use_cov)
	//		pheno_i =pheno_prime.block(0,i,g.Nindv, 1); 
	//	MatrixXdr Xy_i = Xy.block(0, i, g.Nsnp, 1); 
	//	MatrixXdr pheno_sum2 = pheno_i.transpose() *pheno_i;
	//	double pheno_variance = pheno_sum2(0,0) / (g.Nindv-1); 	
	//	compute_se(Xy_i,pheno_i,se, vg,ve,tr2,B,exist_ind);
	//	cout<<"phenotype variance: "<<pheno_variance<<endl; 
	//	cout<<"sigma_g SE: "<<se<<endl; 
	//	cout<<"h2g SE:"<<se/pheno_variance<<endl;
	//	}  
		if(!reg){
			MatrixXdr K = geno_matrix.transpose()* geno_matrix/p- MatrixXdr::Identity(n,n);
			MatrixXdr C= herit(0,0)* geno_matrix.transpose()*geno_matrix /p + herit(1,0)*MatrixXdr::Identity(n,n); 
			MatrixXdr temp = C*K* C *K; 
			MatrixXdr temp1 = pheno * pheno.transpose() * K * C * K; 
			double result=0; 
			double result1=0; 
			for(int i=0; i<n; i++){
				result += temp(i,i);
				result1 += temp1(i,i); 
			}
			result = result*2 +  tr2/100*herit(0,0)*herit(0,0);
			result = sqrt(result); 
			result = result / (tr2-g.Nindv);  
			result1 = result1*2 + tr2/100*herit(0,0)*herit(0,0); 
			result1 = sqrt(result1); 	
			result1 = result1 / (tr2 - g.Nindv); 
			cout<<"no random: "<<result<<endl; 
			cout<<"approximate: "<<result1<<endl; 
		}
	}
//	if(pheno_num>1){
//		cout<<"Coheritability factor estimation: " <<endl; 
//		MatrixXdr b(2,1); 
//		b(0,0)= yKy(0,1);
//		b(1,0)= yy(0,1);
//		MatrixXdr herit  = A.colPivHouseholderQr().solve(b); 
//		cout<<"rho_g: "<<herit(0,0)<<endl;
//		cout<<"rho_e: "<<herit(1,0)<<endl; 
//		cout<<"Genetic Correlation: "<<herit(0,0)/sqrt(VarComp(0,0))/sqrt(VarComp(1,0))<<endl;
//		cout<<"Env Correlation: "<<herit(1,0)/sqrt(1-VarComp(0,0))/sqrt(1-VarComp(1,0))<<endl;
//			
//	} 
	if(gwas){
		//normalize phenotype
		std::string filename=command_line_opts.PHENOTYPE_FILE_PATH; 
		int pheno_sum= read_pheno2(g.Nindv, filename,pheno_idx); 
		MatrixXdr ysum= pheno.colwise().sum(); 
		double phenosum=ysum(0,0); 
		MatrixXdr p_i = ysum/(g.Nindv-1);
		double phenomean= p_i(0,0);  
		MatrixXdr ysum2 = pheno.transpose() * pheno; 
		double phenosum2 = ysum2(0,0); 
		double std = sqrt((phenosum2+g.Nindv*phenomean*phenomean - 2*phenosum*phenomean)/(g.Nindv-1)); 
		for(int i=0; i<g.Nindv; i++)
		{
			pheno(i,0) = pheno(i,0)-phenomean; 
			pheno(i,0) = pheno(i,0)/std; 			

		}
		cout<<"Performing GWAS..."<<endl; 
		//perform per chromsome
		FILE *fp; 
		fp=fopen((filename+".gwas").c_str(), "w");
		fprintf(fp, "%s  %s \n", "STAT", "P_VAL");
		for(int i=0; i<22; i++)
		{
			MatrixXdr geno_mask= MatrixXdr::Zero(g.Nsnp, 1); 
			int block_snp_num=g.get_chrom_snp(i); 
			if(block_snp_num==0)
			{
				cout<<"Chromosome "<<i+1 <<" do not have and SNP"<<endl;
				continue; 
			}
			// Matrix cur is \Sigma = \sigma_g^2 K + \sigma_e^2 I, K is made with the genotype leaving the current chromosome out
			MatrixXdr cur; 
			cur.resize(g.Nsnp-block_snp_num ,n); 
			int left=0; 
			for(int j=0; j<i; j++)
				left+= g.get_chrom_snp(j); 
			int right = left+block_snp_num; 
			if(i==0)
			{
				geno_mask.block(right, 0, g.Nsnp-block_snp_num, 1) = MatrixXdr::Constant(g.Nsnp-block_snp_num, 1, 1); 
				//cur<<geno_matrix.block(right, 0, g.Nsnp-block_snp_num,n); 
			}
			else if(i==22)
			{
				geno_mask.block(left-1,0, g.Nsnp-block_snp_num,1) = MatrixXdr::Constant(g.Nsnp-block_snp_num, 1,1); 
				//cur<<geno_matrix.block(left-1,0, g.Nsnp-block_snp_num,n); 
			}
			else
			{
				geno_mask.block(0,0,left,1) = MatrixXdr::Constant(left, 1,1); 
				geno_mask.block(right, 0, g.Nsnp-left-block_snp_num, 1) = MatrixXdr::Constant(g.Nsnp-left-block_snp_num, 1,1); 
			}	//cur<< geno_matrix.block(0,0,left,n), geno_matrix.block(right,0, g.Nsnp-left-block_snp_num,n); 	
	//		MatrixXdr curInv = cur.transpose()*cur / (g.Nsnp-block_snp_num); 
	//		curInv = curInv*vg; 
	//		for(int j=0; j<n; j++)
	//			curInv(j,j)= curInv(j,j)+ve; 
			
	//		MatrixXdr V = curInv; 
		//	curInv = curInv.inverse();
			MatrixXdr conj_result1(g.Nindv, 1); 
			conjugate_gradient(g.Nindv, vg, ve, geno_mask, pheno, conj_result1, exist_ind);  
			for(int k=0; k<g.get_chrom_snp(i); k++)
			{
				MatrixXdr x_test_mask = MatrixXdr::Zero(1, g.Nsnp); 
				x_test_mask(left+k, 0)=1; 
				MatrixXdr x_test(g.Nindv, 1);
				multiply_y_post(x_test_mask, 1, x_test, false,exist_ind);  
				MatrixXdr resid =MatrixXdr::Constant(g.Nindv, 1,1); 
				x_test  = x_test - resid* means(left+k,0);
				x_test = x_test * stds(left+k, 0);  
					
		//		MatrixXdr x_test = geno_matrix.block(left+k,0,1,n);
		//		MatrixXdr x_test_t = x_test.transpose();  
				//cout<<x_test<<endl; 
				//MatrixXdr conj_gra_result(g.Nindv, 1); 
			//	conjugate_gradient(g.Nindv,vg, ve, V, pheno, conj_gra_result); 
			//	cout<<"conjugate gradient: "<<conj_gra_result(0,0)<<endl; 
			//	MatrixXdr exact_result = curInv*pheno; 
			//	cout<<"exact: "<<exact_result(0,0)<<endl; 
			//      MatrixXdr temp = x_test*curInv*pheno; 
				MatrixXdr temp = x_test * conj_result1; 
				MatrixXdr conj_result2(g.Nindv, 1); 
				conjugate_gradient(g.Nindv, vg, ve, geno_mask, x_test ,conj_result2, exist_ind); 
				
				double temp1 = temp(0,0)*temp(0,0); 
	//			MatrixXdr temp2=x_test*curInv*x_test.transpose(); 
				MatrixXdr temp2 = x_test * conj_result2; 
	//			cout<<"snpts for snp "<<left+k+1<<" :" <<temp1/temp2(0,0)<<endl; 			 
				boost::math::chi_squared mydist(2); 
				float p = boost::math::cdf(mydist, temp1/temp2(0,0)); 
//				cout<<"stat: "<<1-p <<endl
				fprintf(fp, "%f %f \n",temp1/temp2(0,0), 1-p); 
			}
		}
		fclose(fp);
	} 
	//clock_t it_end = clock();
	
		
	//clock_t total_end = clock();
//	double io_time = double(io_end - io_begin) / CLOCKS_PER_SEC;
//	double avg_it_time = double(it_end - it_begin) / (B * 1.0 * CLOCKS_PER_SEC);
//	double total_time = double(total_end - total_begin) / CLOCKS_PER_SEC;
//	cout<<"IO Time:  "<< io_time << "\nAVG Iteration Time:  "<<avg_it_time<<"\nTotal runtime:   "<<total_time<<endl;

	delete[] sum_op;
	delete[] partialsums;
	delete[] yint_e; 
	delete[] yint_m;

	for (int i  = 0 ; i < hsegsize; i++)
		delete[] y_m [i]; 
	delete[] y_m;

	for (int i  = 0 ; i < g.Nindv; i++)
		delete[] y_e[i]; 
	delete[] y_e;

	return 0;
}
