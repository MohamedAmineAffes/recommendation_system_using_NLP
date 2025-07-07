# 🚀 Flask Application with Terraform & GitHub Actions CI/CD

This project demonstrates a complete DevOps pipeline for a simple **Flask application**, using:

👉 **Terraform** for infrastructure provisioning (Local Docker or AWS)\
👉 **Docker** to containerize the Flask application\
👉 **GitHub Actions** to automate testing, building, and deployment\
👉 **CI/CD separation** for Dev (auto-deploy) and Prod (manual approval)

---

## 📦 Project Structure

```
flask-terraform-cicd/
├── app/                # Flask application code
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── infra/              # Terraform infrastructure code
│   ├── main.tf
│   ├── providers.tf
│   └── variables.tf
├── .github/            # GitHub Actions CI/CD workflows
│   └── workflows/
│       ├── dev.yml
│       └── prod.yml
├── README.md
```

---

## ⚙️ Tech Stack

- **Flask** — Python Web Application
- **Docker** — Containerization
- **Terraform** — Infrastructure as Code (Local Docker or AWS)
- **GitHub Actions** — CI/CD Pipeline Automation

---

## 🌐 Application

A basic Flask app with one route:

```python
@app.route("/")
def hello():
    return "Hello from Flask CI/CD!"
```

---

## 🚀 CI/CD Pipeline Overview

| Branch / Trigger     | Action                                                  |
| -------------------- | ------------------------------------------------------- |
| `dev` branch push    | Run tests, build Docker image, Terraform apply (Dev)    |
| `main` branch manual | Terraform plan, manual approval, Terraform apply (Prod) |

---

## ✅ Development Workflow (Local)

1. Ensure **Docker** and **Terraform** are installed.
2. Navigate to `infra` folder.
3. Run:

```bash
terraform init
terraform apply -auto-approve
```

4. Access the app at [http://localhost:5000](http://localhost:5000).

---

## ☁️ Optional AWS Deployment (Free Tier)

To deploy on AWS EC2 (Free Tier):

- Configure Terraform with your AWS credentials.
- Update `infra/main.tf` to provision EC2, install Docker, and run the Flask container.
- Use GitHub Actions for automated deployment to AWS.

**Note**: Requires valid AWS account with Free Tier eligibility.

---

## 🔒 GitHub Secrets

For AWS integration, store these in your GitHub repository:

| Secret Name             | Description         |
| ----------------------- | ------------------- |
| `AWS_ACCESS_KEY_ID`     | Your AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS Secret Key |

---

## ⚡ Example GitHub Actions Workflows

- `.github/workflows/dev.yml` — Automatic deploy for Dev
- `.github/workflows/prod.yml` — Manual approval for Prod

---

## 💡 Future Improvements

- Extend Terraform for databases or load balancers
- Add unit tests with `pytest`
- Integrate Slack or email notifications
- Container registry integration (Docker Hub, GitHub Packages)

---

## 🛠️ Requirements

- [Terraform](https://developer.hashicorp.com/terraform/downloads)
- [Docker](https://docs.docker.com/get-docker/)
- [GitHub Account](https://github.com/)

---

## 👨‍💻 Author

**Your Name** — DevOps Engineer | Cloud Enthusiast

---

## 📢 License

This project is open source and available under the [MIT License](LICENSE).

```
```
