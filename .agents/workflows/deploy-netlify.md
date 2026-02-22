# Deploying ReviewNet to Netlify

ReviewNet's intelligent dashboard is a standard Next.js application and can be easily deployed to Netlify. Because the app is located in a subdirectory (`reviewnet/`), you need to configure your build settings correctly.

## Prerequisites
1. A [Netlify](https://www.netlify.com/) account.
2. Your ReviewNet project pushed to a GitHub, GitLab, or Bitbucket repository.

## Step-by-Step Deployment

### 1. Import Your Project
- Log in to your Netlify dashboard.
- Click **"Add new site"** -> **"Import an existing project"**.
- Connect your git provider and select the **reviewnet** repository.

### 2. Configure Build Settings
This is the most critical step since the Next.js app is not at the root of your repository.

- **Base directory**: `reviewnet`
- **Build command**: `npm run build`
- **Publish directory**: `.next` (Netlify should auto-detect this once the base directory is set, but verify it if asked).

### 3. Environment Variables (Optional)
If you have any API keys or custom environment variables, add them in **Site configuration** -> **Environment variables**.

### 4. Deploy
Click **"Deploy reviewnet"**. Netlify will install dependencies, build the Next.js app, and provide you with a live URL.

## Automatic Deployments
Netlify will automatically trigger a new build every time you push changes to your repository.

> [!TIP]
> Netlify automatically installs the **Next.js Runtime**, which handles server-side rendering (SSR) and API routes for you. You don't need any extra configuration for this to work.
