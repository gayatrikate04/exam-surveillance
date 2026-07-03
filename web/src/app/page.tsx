import { ExamMonitor } from "@/components/exam-monitor";
import { HomeDashboard } from "@/components/home-dashboard";
import { sessionIsAdmin } from "@/lib/admin-role";
import { getSessionFromCookies } from "@/lib/session";

export default async function Home() {
  const session = await getSessionFromCookies();

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <HomeDashboard isAdmin={session ? sessionIsAdmin(session) : false} email={session?.email ?? null}>
        <ExamMonitor />
      </HomeDashboard>
    </div>
  );
}
