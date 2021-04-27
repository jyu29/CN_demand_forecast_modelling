node {
    script {
        echo "Launching modeling for cutoff(s) '$list_cutoff' with run name '${run_name}'..."
    }
}

pipeline {
    agent any
    environment {
        gitUrl = 'https://github.com/dktunited/forecast-modeling-demand/'
    }
    stages {
        stage('Demand Forecast Modeling') {
            agent {
              label 'PYLINUXT3XLARGE'
            }
            
            environment {
                conda_env = 'forecast-modeling-demand'
            }
            steps {
                git changelog: false, credentialsId: 'github_dktjenkins', poll: false, url: "${gitUrl}", branch: "${branch_name}"
                sh('''
                source ~/miniconda3/etc/profile.d/conda.sh
                conda env update -f environment.yml
                conda run --no-capture-output -n ${conda_env} python -u main.py
                ''')
            }
        }
    }

    post {
            failure {
                    mail to: 'forecastunited@decathlon.net',
                    subject: "Pipeline ${env.JOB_NAME} failed", body: "${env.BUILD_URL}"
                    }

            unstable {
                    mail to: 'forecastunited@decathlon.net',
                    subject: "Pipeline ${env.JOB_NAME} unstable", body: "${env.BUILD_URL}"
                    }
        }
}

