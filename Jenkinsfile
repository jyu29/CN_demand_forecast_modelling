pipeline {
    agent any
    environment {
        branch = 'develop'
        gitUrl = 'https://github.com/dktunited/forecast-modeling-demand.git'
        run_env = "${runenvinjob}"
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
                    conda create -n fcst_modeling_demand python=3
                    conda init /bin/sh
                    conda activate fcst_modeling_demand
                    python3 main.py --environment ${run_env} --only_last ${only_last}
                    conda deactivate
                    ''')
                }
        }
    }
    
    post {
            failure { 
                    mail to: 'ouiame.aitelkadi@decathlon.com',
                    subject: '[FAILED] Pipeline Demand Forecast has failed', body: "${env.BUILD_URL}"
                    }
        
            unstable {
                    mail to: 'ouiame.aitelkadi@decathlon.com',
                    subject: '[UNSTABLE] Pipeline Demand Forecast is unstable', body: "${env.BUILD_URL}"
                    }
        }
}
