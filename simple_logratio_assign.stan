functions{
  real[,,] assign_cvals (int n_areas, int R, int C, matrix row_margins, matrix col_margins, real[,,] lambda){
     matrix[n_areas, R] slack_row;
     matrix[n_areas, C] slack_col;
     real cell_value[n_areas, R, C];
     vector[2] lower_pos;
     vector[2] upper_pos;
     real lower_bound;
     real upper_bound;
     real rt;
     
     slack_row=row_margins;
     slack_col=col_margins;

     lower_pos[1]=0.0;

     for (j in 1:n_areas){
            rt=sum(row(slack_row, j));
            for (c in 1:(C-1)){
              for (r in 1:(R-1)){
                if (r ==1){
                  lower_pos[2]=slack_col[j, c]+slack_row[j, r]-rt;
                  lower_bound=max(lower_pos);
                  } else {
                    lower_pos[2]=slack_col[j, c]-sum(tail(row(slack_row, j), R-r));
                    lower_bound=max(lower_pos);
                    }
                    upper_pos[1]=slack_col[j, c];
                    upper_pos[2]=slack_row[j, r];
                    upper_bound=min(upper_pos);
                    cell_value[j, r,c]=lower_bound+inv_logit(lambda[j,r,c])*(upper_bound-lower_bound);
                    slack_col[j, c]=slack_col[j, c]-cell_value[j, r,c];
                    slack_row[j, r]=slack_row[j, r]-cell_value[j, r,c];
                    rt=rt-cell_value[j, r,c];
                    }
                    cell_value[j, R, c]=slack_col[j, c];
                    rt=rt-cell_value[j, R, c];
                    slack_col[j, c]=slack_col[j, c]-cell_value[j, R, c];
                    slack_row[j, R]=slack_row[j, R]-cell_value[j, R, c];
                    }
                    for (r in 1:(R-1)){
                      cell_value[j, r, C]=slack_row[j, r];
                      rt=rt-cell_value[j, r, C];
                      slack_col[j, C]=slack_col[j, C]-cell_value[j, r, C];
                      slack_row[j, r]=slack_row[j, r]-cell_value[j, r, C];
                      }
                      cell_value[j, R, C]=rt;
                      }
                      return cell_value;
  }
}

data{
 int<lower=0> n_areas;
 int<lower=0> R;
 int<lower=0> C;
 matrix<lower=0>[n_areas, R] row_margins;
 matrix<lower=0>[n_areas, C] col_margins;
}
transformed data{
 matrix[n_areas, R] row_margins_prop;
 matrix[n_areas, C] col_margins_prop;

 for (j in 1:n_areas){
  for (r in 1:R){
   row_margins_prop[j, r]=row_margins[j,r]/sum(row(row_margins, j));
  }
  for (c in 1:C){
   col_margins_prop[j, c]=(col_margins[j,c])/sum(row(col_margins, j));
  }
 }
}

parameters{
 vector[(R-1)*(C-1)] lambda_2[n_areas];
 #vector<lower=0>[(R-1)*C] sd_crdiff;
 #real<lower=0> sd_clr;
 vector<lower=0.001>[R*(C-1)] sd_clr;
 #vector<lower=0>[R*(C-1)] sd_noise;
 real<lower=0> sd_noise;
 #corr_matrix[R] L_rmdiff;
 #vector<lower=0>[C] sd_cm;
 #vector<lower=0>[(R-1)*C] sd_crdiff;
 simplex[C] beta_d[R];
 real<lower=0> A;
 real<lower=0> mu_noise;
 vector[R*(C-1)] cell_logratio_noise[n_areas];
 #real<lower=0> mu_sd_clr;
}

transformed parameters{
 matrix[n_areas, C] col_margins_expect;
 vector[C] col_margins_errors[n_areas];
 #matrix[C, C] S_cm;
 matrix[(R-1)*C, (R-1)*C] S_clr;
 matrix[R, C] betas; 
 #matrix[R, R] Sigma_rmdiff;
 real lambda[n_areas, R,C];
 real cell_value[n_areas, R, C];
 real cell_logratio_matrix[n_areas, R, C-1];
 real cell_logratio_expect[n_areas, R, C-1];
 vector[R * (C-1)] cell_logratio_errors[n_areas];
 
 
 
 for (j in 1:n_areas){
  for (r in 1:(R-1)){
    for (c in 1:(C-1))
   lambda[j, r, c]=lambda_2[j, (r-1)*(C-1)+c];
  }
 }

 for (r in 1:R){
   for (c in 1:C){
      betas[r,c]=beta_d[r,c];
   }
 }
 
 cell_value=assign_cvals(n_areas, R, C, row_margins, col_margins, lambda);
 
  for (j in 1:n_areas){
   for (r in 1:R){
    for (c in 1:(C-1)){
     cell_logratio_matrix[j, r , c]=log((cell_value[j,r,c]+1)/(cell_value[j, r, C]+1));
    }
   }
  }

  
 #S_cm=diag_matrix(sd_cm);
 S_clr=diag_matrix(sd_clr);
 #S_clr=diag_matrix(rep_vector(sd_clr, R*(C-1)));
 #Sigma_rmdiff=S_rmdiff*L_rmdiff*S_rmdiff;
  for (j in 1:n_areas){
    for (r in 1:R){
      for (c in 1:C-1){
        cell_logratio_expect[j, r, c]=log(betas[r, c]/betas[r,C]);
        cell_logratio_errors[j, (c-1)*R + r]=cell_logratio_matrix[j, r, c]-cell_logratio_expect[j, r, c]+cell_logratio_noise[j, (c-1)*R + r];

      }
    }
  }
for (j in 1:n_areas){
    for (c in 1:C){
        col_margins_expect[j, c]=dot_product(col(betas,c), row(row_margins_prop, j));
        col_margins_errors[j, c]=(col_margins_prop[j, c]-col_margins_expect[j, c]);
    }
  }
}
model{
  lambda_2~multi_normal(rep_vector(0, (R-1)*(C-1)), diag_matrix(rep_vector(10, (R-1)*(C-1))));
  #col_margins_errors~multi_normal(rep_vector(0, C), S_cm);
  cell_logratio_noise~multi_normal(rep_vector(0, R*(C-1)), diag_matrix(rep_vector(sd_noise, R*(C-1))));
  rep_vector(0, R*(C-1))~multi_normal(cell_logratio_errors, S_clr);
  #sd_cm~cauchy(0, 25);
  sd_noise~cauchy(mu_noise, A);
}
