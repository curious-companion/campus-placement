import { createBrowserRouter, Navigate, RouterProvider } from 'react-router-dom'
import { Home } from './pages/home'
import Campus from './pages/campus'
import { PAGE_PATHS } from './constants/PagePaths'
import { AppContextProvider } from './contexts/AppContext'
import { PageNotFound } from './pages/notfound'
import { Student } from './pages/student'
import  ResumeUpload  from './pages/resume/resume.jsx'; // ✅ Already imported
import { Analytics } from '@vercel/analytics/react'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Navigate to={PAGE_PATHS.INSIGHTS} />,
  },
  {
    path: PAGE_PATHS.INSIGHTS,
    element: <Home />,
  },
  {
    path: PAGE_PATHS.CAMPUS_PLACEMENT_ANALYZER,
    element: <Campus />,
  },
  {
    path: PAGE_PATHS.STUDENT_PLACEMENT_ANALYZER,
    element: <Student />,
  },
  {
    path: PAGE_PATHS.ATS_RESUME_CHECKER, // ✅ New route
    element: <ResumeUpload />,
  },
  {
    path: '*',
    element: <PageNotFound />,
  },
])

function App() {
  return (
    <AppContextProvider>
      <RouterProvider router={router} />
      <Analytics />
    </AppContextProvider>
  )
}

export default App
