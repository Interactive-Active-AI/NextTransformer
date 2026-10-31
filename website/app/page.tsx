import {baselineIdentity} from './lib/baseline';
import Dashboard from './dashboard';
import {methods,runs,updatedAt,SOURCE} from './lib/results';
const initial={schemaVersion:1,updatedAt,source:SOURCE,runs:runs.map(baselineIdentity)};
export default function Home(){return <Dashboard initial={initial as any} initialMethods={methods as any}/>;}
