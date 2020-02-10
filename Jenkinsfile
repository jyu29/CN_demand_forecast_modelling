pipeline {
    agent any
    environment {
        run_env = "${run_env}"
        only_last = "${only_last}"
    }
    stages {
        stage('Prepare data and make forecasts') {
            environment {
                HTTPS_PROXY="http://proxy-internet-aws-eu.subsidia.org:3128"
                HTTP_PROXY="http://proxy-internet-aws-eu.subsidia.org:3128"
            }
            steps {
                sh('''
                    if [ -d /var/lib/jenkins/.conda/envs/fcst_modeling_demand ]; then rm -rf /var/lib/jenkins/.conda/envs/fcst_modeling_demand; fi
                    conda create -n fcst_modeling_demand python=3.6.8
                    CONDA_BASE=$(conda info --base)
                    source $CONDA_BASE/etc/profile.d/conda.sh
                    conda activate fcst_modeling_demand
                    pip install -r requirements.txt
                    python main.py --environment ${run_env} --only_last ${only_last}
                    conda deactivate
                    ''')
                }
        }
    }
    
    post {
            success {
                    mail to: 'forecastunited@decathlon.net',
                    subject: '[SUCCESS] Demand Forecast RUN Pipeline has finished successfully', body: "${env.BUILD_URL}"
                   }

            failure {
                    mail to: 'forecastunited@decathlon.net',
                    subject: '[FAILED] Demand Forecast RUN Pipeline has failed', body: "${env.BUILD_URL}"
                   }

            unstable {
                    mail to: 'forecastunited@decathlon.net',
                    subject: '[UNSTABLE] Demand Forecast RUN Pipeline is unstable', body: "${env.BUILD_URL}"
                   }
       }
}
