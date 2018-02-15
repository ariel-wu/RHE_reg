/** 
 All of this code is written by Aman Agrawal 
 (Indian Institute of Technology, Delhi)
*/
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
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

clock_t total_begin = clock();
MatrixXdr pheno; 
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


options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize=false;
int accelerated_em=0;
double convergence_limit;
bool memory_efficient = false;
bool missing=false;
bool fast_mode = true;
bool text_version = false;

void read_pheno(int Nind, std::string filename){
	pheno.resize(Nind, 1); 
	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	int i=0;  
	while(std::getline(ifs, line)){
		pheno(i,0) = atof(line.c_str());
		i++;  
	}

}
void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
	
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
				res(p_iter,k_iter) = res(p_iter,k_iter)/(g.get_col_std(p_iter));		
 		}		
 	}	

}

void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res,bool subtract_means){

	MatrixXdr op;
	op = op_orig.transpose();

	if(var_normalize && subtract_means){
		for(int p_iter=0;p_iter<p;p_iter++){
			for(int k_iter=0;k_iter<Nrows_op;k_iter++)		
				op(p_iter,k_iter) = op(p_iter,k_iter) / (g.get_col_std(p_iter));		
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

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	for(int p_iter=0;p_iter<p;p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++){
			double temp=0;
			for(int n_iter=0;n_iter<n;n_iter++)
				temp+= g.get_geno(p_iter,n_iter,var_normalize)*op(n_iter,k_iter);
			res(p_iter,k_iter)=temp;
		}
	}
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	for(int n_iter=0;n_iter<n;n_iter++){
		for(int k_iter=0;k_iter<Nrows_op;k_iter++){
			double temp=0;
			for(int p_iter=0;p_iter<p;p_iter++)
				temp+= op(k_iter,p_iter)*(g.get_geno(p_iter,n_iter,var_normalize));
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

void multiply_y_post(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_post_fast(op,Nrows_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_post_naive_mem(op,Nrows_op,res);
		else
			multiply_y_post_naive(op,Nrows_op,res);
	}
}

void multiply_y_pre(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_pre_fast(op,Ncol_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_pre_naive_mem(op,Ncol_op,res);
		else
			multiply_y_pre_naive(op,Ncol_op,res);
	}
}

pair<double,double> get_error_norm(MatrixXdr &c){
	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(p,k);
	MatrixXdr q_t(k,p);
	q_t = Q.transpose();
	MatrixXdr b(k,n);
	multiply_y_post(q_t,k,b,true);
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
                e_l(p_iter,n_iter) = g.get_geno(p_iter,n_iter,var_normalize) - b_l(p_iter,n_iter);
                e_k(p_iter,n_iter) = g.get_geno(p_iter,n_iter,var_normalize) - b_k(p_iter,n_iter);
            }
        }

        double ek_norm = e_k.norm();
        double el_norm = e_l.norm();
        return make_pair(ek_norm,el_norm);
    }
}

MatrixXdr run_EM_not_missing(MatrixXdr &c_orig){
	
	#if DEBUG==1
		if(debug){
			print_time ();
			cout << "Enter: run_EM_not_missing" << endl;
		}
	#endif

	MatrixXdr c_temp(k,p);
	MatrixXdr c_new(p,k);
	c_temp = ( (c_orig.transpose()*c_orig).inverse() ) * (c_orig.transpose());
	
	#if DEBUG==1
		if(debug){
			print_timenl ();
		}
	#endif
	
	MatrixXdr x_fn(k,n);
	multiply_y_post(c_temp,k,x_fn,true);
	
	#if DEBUG==1
		if(debug){
			print_timenl ();
		}
	#endif
	
	MatrixXdr x_temp(n,k);
	x_temp = (x_fn.transpose()) * ((x_fn*(x_fn.transpose())).inverse());
	multiply_y_pre(x_temp,k,c_new,true);
	
	#if DEBUG==1
		if(debug){
			print_time ();
			cout << "Exiting: run_EM_not_missing" << endl;
		}
	#endif

	return c_new;
}

MatrixXdr run_EM_missing(MatrixXdr &c_orig){
	
	MatrixXdr c_new(p,k);

	MatrixXdr mu(k,n);
	
	// E step
	MatrixXdr c_temp(k,k);
	c_temp = c_orig.transpose() * c_orig;

	MatrixXdr T(k,n);
	MatrixXdr c_fn;
	c_fn = c_orig.transpose();
	multiply_y_post(c_fn,k,T,false);

	MatrixXdr M_temp(k,1);
	M_temp = c_orig.transpose() *  means;
	
	for(int j=0;j<n;j++){
		MatrixXdr D(k,k);
		MatrixXdr M_to_remove(k,1);
		D = MatrixXdr::Zero(k,k);
		M_to_remove = MatrixXdr::Zero(k,1);
		for(int i=0;i<g.not_O_j[j].size();i++){
			int idx = g.not_O_j[j][i];
			D = D + (c_orig.row(idx).transpose() * c_orig.row(idx));
			M_to_remove = M_to_remove + (c_orig.row(idx).transpose()*g.get_col_mean(idx));
		}
		mu.col(j) = (c_temp-D).inverse() * ( T.col(j) - M_temp + M_to_remove);
	}

	// M step

	MatrixXdr mu_temp(k,k);
	mu_temp = mu * mu.transpose();
	MatrixXdr T1(p,k);
	MatrixXdr mu_fn;
	mu_fn = mu.transpose();
	multiply_y_pre(mu_fn,k,T1,false);
	MatrixXdr mu_sum(k,1);
	mu_sum = MatrixXdr::Zero(k,1);
	mu_sum = mu.rowwise().sum();

	for(int i=0;i<p;i++){
		MatrixXdr D(k,k);
		MatrixXdr mu_to_remove(k,1);
		D = MatrixXdr::Zero(k,k);
		mu_to_remove = MatrixXdr::Zero(k,1);
		for(int j=0;j<g.not_O_i[i].size();j++){
			int idx = g.not_O_i[i][j];
			D = D + (mu.col(idx) * mu.col(idx).transpose());
			mu_to_remove = mu_to_remove + (mu.col(idx));
		}
		c_new.row(i) = (((mu_temp-D).inverse()) * (T1.row(i).transpose() -  ( g.get_col_mean(i) * (mu_sum-mu_to_remove)))).transpose();
		double mean;
		mean = g.get_col_sum(i);
		mean = mean -  (c_orig.row(i)*(mu_sum-mu_to_remove))(0,0);
		mean = mean * 1.0 / (n-g.not_O_i[i].size());
		g.update_col_mean(i,mean);
	}
	return c_new;
}

MatrixXdr run_EM(MatrixXdr &c_orig){
	
	if(missing)
		return run_EM_missing(c_orig);
	else
		return run_EM_not_missing(c_orig);
}

void print_vals(){

	HouseholderQR<MatrixXdr> qr(c);
	MatrixXdr Q;
	Q = qr.householderQ() * MatrixXdr::Identity(p,k);
	MatrixXdr q_t(k,p);
	q_t = Q.transpose();
	MatrixXdr b(k,n);
	multiply_y_post(q_t,k,b,true);
	JacobiSVD<MatrixXdr> b_svd(b, ComputeThinU | ComputeThinV);
	MatrixXdr u_l; 
	u_l = b_svd.matrixU();
	MatrixXdr v_l;
	v_l = b_svd.matrixV();
	MatrixXdr u_k;
	MatrixXdr v_k,d_k;
	u_k = u_l.leftCols(k_orig);
	v_k = v_l.leftCols(k_orig);

	ofstream evec_file;
	evec_file.open((string(command_line_opts.OUTPUT_PATH)+string("evecs.txt")).c_str());
	evec_file<< std::setprecision(15) << Q*u_k << endl;
	evec_file.close();
	ofstream eval_file;
	eval_file.open((string(command_line_opts.OUTPUT_PATH)+string("evals.txt")).c_str());
	for(int kk =0 ; kk < k_orig ; kk++)
		eval_file << std::setprecision(15)<< (b_svd.singularValues())(kk)<<endl;
	eval_file.close();

	ofstream proj_file;
	proj_file.open((string(command_line_opts.OUTPUT_PATH) + string("projections.txt")).c_str());
	proj_file << std::setprecision(15)<< v_k<<endl;
	proj_file.close();
	if(debug){
		ofstream c_file;
		c_file.open((string(command_line_opts.OUTPUT_PATH)+string("cvals.txt")).c_str());
		c_file<<c<<endl;
		c_file.close();
		
		d_k = MatrixXdr::Zero(k_orig,k_orig);
		for(int kk =0 ; kk < k_orig ; kk++)
			d_k(kk,kk)  =(b_svd.singularValues())(kk);
		MatrixXdr x_k;
		x_k = d_k * (v_k.transpose());
		ofstream x_file;
		x_file.open((string(command_line_opts.OUTPUT_PATH) + string("xvals.txt")).c_str());
		x_file<<x_k.transpose()<<endl;
		x_file.close();
	}
}

int main(int argc, char const *argv[]){

	clock_t io_begin = clock();
    clock_gettime (CLOCK_REALTIME, &t0);

	pair<double,double> prev_error = make_pair(0.0,0.0);
	double prevnll=0.0;

	parse_args(argc,argv);

	
	//TODO: Memory effecient Version of Mailman

	memory_efficient = command_line_opts.memory_efficient;
	text_version = command_line_opts.text_version;
	fast_mode = command_line_opts.fast_mode;
	missing = command_line_opts.missing;

	
	if(text_version){
		if(fast_mode)
			g.read_txt_mailman(command_line_opts.GENOTYPE_FILE_PATH,missing);
		else
			g.read_txt_naive(command_line_opts.GENOTYPE_FILE_PATH,missing);
	}
	else{
		g.read_plink(command_line_opts.GENOTYPE_FILE_PATH,missing,fast_mode);
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

	int B = command_line_opts.batchNum; 
	k_orig = command_line_opts.num_of_evec ;
	debug = command_line_opts.debugmode ;
	check_accuracy = command_line_opts.getaccuracy;
	var_normalize = false; 
	accelerated_em = command_line_opts.accelerated_em;
	k = k_orig + command_line_opts.l;
	k = (int)ceil(k/10.0)*10;
	command_line_opts.l = k - k_orig;
	p = g.Nsnp;
	n = g.Nindv;
	convergence_limit = command_line_opts.convergence_limit;
	bool toStop=false;
	if(convergence_limit!=-1)
		toStop=true;
	srand((unsigned int) time(0));
	c.resize(p,k);
	x.resize(k,n);
	v.resize(p,k);
	means.resize(p,1);
	stds.resize(p,1);
	sum2.resize(p,1); 
	sum.resize(p,1); 

	if(!fast_mode && !memory_efficient){
		geno_matrix.resize(p,n);
		g.generate_eigen_geno(geno_matrix,var_normalize);
	}
	
	clock_t io_end = clock();

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
		stds(i,0) =1/ g.get_col_std(i);
		sum2(i,0) =g.get_col_sum2(i); 
		sum(i,0)= g.get_col_sum(i); 
	}



//	cout<<"printing means: "<<endl<<means<<endl; 
//	cout<<"printing std: "<<endl<<stds<<endl; 	
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


	//get geno
	//cout<<g.get_geno(0,0,false);
	//read phenotype
	//
	//
	std::string filename=command_line_opts.PHENOTYPE_FILE_PATH; 
	read_pheno(g.Nindv, filename); 
	double y_sum=pheno.sum(); 
	double y_mean = y_sum/g.Nindv;
	for(int i=0; i<g.Nindv; i++) 
		pheno(i,0) =pheno(i,0) - y_mean; //center phenotype	
	y_sum=pheno.sum(); 
	//correctness check 
	//MatrixXdr zb = MatrixXdr::Random(1, g.Nsnp); 
	//MatrixXdr res(1,g.Nindv); 
	//multiply_y_post_fast(zb, 1, res, false); 
	//cout<< MatrixXdr::Constant(1,4,1)<<endl;  
	
	//compute y^TKy
	MatrixXdr res(g.Nsnp, 1); 
	multiply_y_pre_fast(pheno,1,res,false); 
	res = res.cwiseProduct(stds); 
	MatrixXdr resid(g.Nsnp, 1); 
	resid = means.cwiseProduct(stds); 
	resid = resid *y_sum; 	
	MatrixXdr Xy(g.Nsnp,1); 
	Xy = res-resid; 
	double yKy = (Xy.array()* Xy.array()).sum(); 
	yKy = yKy/g.Nsnp; 
//	cout<< "yKy: "<<yKy<<endl;  
	//compute tr[K]
	double tr_k =0 ;
	MatrixXdr temp = sum2 + g.Nindv* means.cwiseProduct(means) - 2 * means.cwiseProduct(sum);
	temp = temp.cwiseProduct(stds);
	temp = temp.cwiseProduct(stds); 
	tr_k = temp.sum() / g.Nsnp;
//	clock_t trK =clock(); 
//	cout<<"computing trace of K: "<<trK-it_begin<<endl;  
//	for (int j=0; j<g.Nsnp; j++)
//	{
//		double temp = sum2(j,0)+g.Nindv*means(j,0)*means(j,0)- 2*means(j,0)*sum(j,0); 
//		temp = temp * stds(j,0) * stds(j,0); 
//		tr_k += temp; 
//	} 	
//	tr_k = tr_k / g.Nsnp; 
	//compute tr[K^2]
	//for gaussian
//	std::random_device rd; 
//	std::mt19937 gen(rd()); 
//	std::normal_distribution<> d(0,1); 

//	boost::normal_distribution<> dist(0,1); 
//	boost::mt19937 gen; 
//	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > random_vec(gen,dist); 
	double tr_k2=0;
	VectorXd vec(g.Nsnp);
	for(int i=0; i<g.Nsnp; i++)
		vec(i)=stds(i); 
	//DiagonalMatrix<double,a> Sigma(a); 
	//Sigma.diagonal()=vec; 
//	cout<<"sigma: "<<Sigma<<endl; 

	//cout<<"vec: "<<vec<<endl; 
	clock_t it_begin =clock(); 
	for(int i=0; i<B; i++){
		//G^T zb 
        	//clock_t random_step=clock(); 
		MatrixXdr zb= MatrixXdr::Random(g.Nindv, 10);
		zb = zb * sqrt(3); 
//		for(int t=0; t<g.Nindv; t++)
//			for(int k=0; k<10; k++)
//				zb(t,k) = random_vec(); 	
		MatrixXdr res(g.Nsnp, 10); 
		//clock_t gen_zb=clock(); 
	//	cout<<"generating random Zb: "<<endl;
	//	print_time(); 
		multiply_y_pre_fast(zb,10,res, false);
	//	cout<<"mailman algorithm "<<endl;
	//	print_time(); 	 
//		cout<<"printing result"<< endl<<res<<endl; 
		//sigma scale \Sigma G^T zb; compute zb column sum
		MatrixXdr zb_sum = zb.colwise().sum(); 
	//	cout<<"column sum of Zb "<<endl; 
	//	print_time(); 
		//cout<<zb_sum<<endl;  
		//std::vector<double> zb_sum(10,0);  
		//res = Sigma*res;
		for(int j=0; j<g.Nsnp; j++)
		        for(int k=0; k<10;k++)
		             res(j,k) = res(j,k)*stds(j,0); 
	//	cout<<"scaling result: "<<endl; 
	//	print_time();  
		//compute /Sigma^T M z_b
//		cout<<"printing Sigma G^T zb: " <<endl<<res<<endl; 
		MatrixXdr resid(g.Nsnp, 10);
		MatrixXdr inter = means.cwiseProduct(stds);
		//cout<<"inter: "<<inter<<endl; 
		resid = inter * zb_sum;
	//	cout<<"scaling residual "<<endl; print_time();    
//		cout<<"printing first resisudual: "<<endl<<resid<<endl; 
		zb = res - resid; // X^Tzb =zb' 
		clock_t Xtzb = clock(); 
	//	cout<<"compute X^T Zb: "<<endl;print_time();  
//		cout<<"printing X^tzb : " <<endl<<zb <<endl;
		//cout<<zb<<endl; 
		//compute zb' %*% /Sigma 
		//zb = Sigma*zb ; 
		
              	for(int k=0; k<10; k++){
                  for(int j=0; j<g.Nsnp;j++){
                        zb(j,k) =zb(j,k) *stds(j,0);}}
                                              
		MatrixXdr new_zb = zb.transpose(); 
		MatrixXdr new_res(10, g.Nindv);
//		cout<<"print zb transpose: "<<endl<<new_zb<<endl; 
		multiply_y_post_fast(new_zb, 10, new_res, false); 
//		cout<< "printing zb^T sigma g^t "<<endl <<new_res<<endl;  
		//new_res =  zb'^T \Sigma G^T 10*N 
		MatrixXdr new_resid(10, g.Nsnp); 
		MatrixXdr zb_scale_sum = new_zb * means;
//		cout<<"printing zb_scale_sum: "<<endl<< zb_scale_sum<<endl; 
		new_resid = zb_scale_sum * MatrixXdr::Constant(1,g.Nindv, 1);  
	//	cout<<"printing new_resid: "<<endl<<new_resid<<endl; 
	//	cout<<((new_res - new_resid).array() * (new_res-new_resid).array()).sum()/g.Nsnp/g.Nsnp/10;  
		tr_k2+= ((new_res - new_resid).array() * (new_res-new_resid).array()).sum();  
//		clock_t rest = clock(); 
//		cout<<"computing rest: "<<rest - Xtzb<<endl; 
		
	}
	tr_k2  = tr_k2 /10/g.Nsnp/g.Nsnp/B; 
//	cout<<"approximated tr[k^2]: "<<tr_k2<<endl;
//	cout<<"tr[k]: "<<tr_k<<endl;

	MatrixXdr A(2,2); 
	A(0,0)=tr_k2; A(0,1)=tr_k; A(1,0)=tr_k; A(1,1)= g.Nindv; 
	MatrixXdr b(2,1); 
	b(0,0) = yKy; 
	b(1,0) = (pheno.array() * pheno.array()).sum(); 
	MatrixXdr herit = A.colPivHouseholderQr().solve(b); 
	cout<<"V(G): "<<herit(0,0)<<endl; 
	cout<<"V(e): "<<herit(1,0)<<endl; 
	cout<<"Vp "<<herit.sum()<<endl; 
	cout<<"V(G)/Vp: "<<herit(0,0)/herit.sum()<<endl;   
	double c = g.Nindv* (tr_k2/g.Nindv - tr_k*tr_k/g.Nindv/g.Nindv); 
	cout<<"SE: "<<sqrt(2/c)<<endl; 
	clock_t it_end = clock();

    print_vals();
		
	clock_t total_end = clock();
	double io_time = double(io_end - io_begin) / CLOCKS_PER_SEC;
	double avg_it_time = double(it_end - it_begin) / (B * 1.0 * CLOCKS_PER_SEC);
	double total_time = double(total_end - total_begin) / CLOCKS_PER_SEC;
	cout<<"IO Time:  "<< io_time << "\nAVG Iteration Time:  "<<avg_it_time<<"\nTotal runtime:   "<<total_time<<endl;

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
