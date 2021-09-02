functions {
  vector f_data_augment(int[] idx_binary, vector v_1, vector v_0) {
    // instantiations
		int N_ = num_elements(idx_binary);
		int i = 1; int o = 1;
		vector[N_] v;
		
		// error check
  // 		if( (num_elements(v_1) + num_elements(v_0)) != N_ ) {
  // 		  print("len(v_1) + len(v_0) != len(idx_binary), but should in f_data_augment |",
  // 		        " | len(v_1) = ", num_elements(v_1),
  // 		        " | len(v_0) = ", num_elements(v_0),
  // 		        " | len(idx_binary) = ", N_);
  //     }
		for(n in 1:N_) {
			if(idx_binary[n] == 1) {
				v[n] = v_1[i];
				i += 1;
			} else {
				v[n] = v_0[o];
				o += 1;
			}
		}
		
		return v;
	}
}

data {
	int<lower=1> N; 				        // no. of total units
	int<upper=N> N_o;               // no. of units with observed outcome
	int<lower=1> K; 				        // no. of covariates
	
	matrix[N, K] X;		              // covariates
	int<lower=0, upper=1> Y_s[N];	  // selection indicator, binary, observed for all N units
	vector[N_o] Y_o;	              // conditional outcome, continuous, observed only for N_o units
}

transformed data {
  int<lower=0> N_pos = sum(Y_s);
	int<lower=0> N_neg = N - N_pos;
}

parameters {
	// regression coefficients
	vector[K] beta_s;
	vector[K] beta_o;
	
	// covariance parameters
	real<lower=0> tau;
	cholesky_factor_corr[2] L_corr;  
	
	// latent variables
	vector<upper=0>[N_neg] Z_s_neg;
	vector<lower=0>[N_pos] Z_s_pos;
	vector[N - N_o] Z_o_neg;
}

transformed parameters {
  // data augmented selection and outcome LHS variates
  vector[N] Z_s = f_data_augment(Y_s, Z_s_pos, Z_s_neg);
	vector[N] Z_o = f_data_augment(Y_s, Y_o, Z_o_neg);
}

model {
  // intermediate objects that will not be saved
  matrix[N, 2] Z = append_col(Z_s, Z_o);
  matrix[N, 2] XB = X * append_col(beta_s, beta_o);
  matrix[2, 2] L_sigma = diag_pre_multiply([1, tau], L_corr);
  
  // increment posterior via '~'
  for(i in 1:N)
    Z[i] ~ multi_normal_cholesky(XB[i], L_sigma);

  beta_s ~ normal(0, 3);
  beta_o ~ normal(0, 3);
  tau ~ cauchy(0, 2);
  L_corr ~ lkj_corr_cholesky(3);
}

generated quantities {
  // real heckman_rho = (L_corr*L_corr')[1, 2];
  real heckman_rho = tcrossprod(to_matrix(L_corr))[1,2];
}
