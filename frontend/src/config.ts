import config from "./config.json" ;

interface IConfig {
    kazuApiUrl: () => string
    authEnabled: () => boolean
}

class Config implements IConfig {
    private _kazuApiURl: string;
    private _authEnabled: boolean;
    kazuApiUrl() {
        return this._kazuApiURl;
    }

    authEnabled() {
        return this._authEnabled;
    }
    constructor() {
        this._kazuApiURl = config["kazuApiUrl"] as string
        this._authEnabled = config["authEnabled"] as boolean
    }
}

export {Config};
export type { IConfig };
