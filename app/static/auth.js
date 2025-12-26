/**
 * Centralized Authentication Logic using Supabase
 */

class AuthManager {
    constructor(supabaseUrl, supabaseKey) {
        this.supabase = null;
        this.mockMode = !supabaseUrl || !supabaseKey;
        this.user = null;
        this.session = null;
        this.onStateChangeCallbacks = [];

        if (!this.mockMode) {
            // Initialize Supabase Client
            // Assumes supabase-js is loaded via CDN
            if (typeof supabase !== 'undefined') {
                this.supabase = supabase.createClient(supabaseUrl, supabaseKey);
                
                // Initialize listener
                this.supabase.auth.onAuthStateChange((event, session) => {
                    this._handleAuthStateChange(event, session);
                });
            } else {
                console.error('Supabase JS client not loaded.');
                this.mockMode = true;
            }
        } else {
            console.log('AuthManager initialized in MOCK MODE');
            // Check for existing mock session in localStorage
            const mockToken = localStorage.getItem('mock_token');
            const mockUserStr = localStorage.getItem('mock_user');
            
            if (mockToken) {
                let user = { email: 'mock@user.com', id: 'mock-user' };
                if (mockUserStr) {
                    try { user = JSON.parse(mockUserStr); } catch(e) {}
                }
                
                this.user = user;
                this.session = { access_token: mockToken, user: user };
                // Defer to allow listeners to register
                setTimeout(() => this._notifyListeners('SIGNED_IN', this.session), 0);
            }
        }
    }

    /**
     * Register a callback for auth state changes.
     * @param {Function} callback - (event, session) => void
     */
    onAuthStateChange(callback) {
        this.onStateChangeCallbacks.push(callback);
    }

    _notifyListeners(event, session) {
        this.onStateChangeCallbacks.forEach(cb => cb(event, session));
    }

    _handleAuthStateChange(event, session) {
        console.log('Auth State Change:', event);
        this.session = session;
        this.user = session ? session.user : null;
        this._notifyListeners(event, session);
    }

    /**
     * Sign Up with Email and Password
     */
    async signUp(email, password) {
        if (this.mockMode) {
            try {
                const res = await fetch('/api/auth/signup', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password})
                });
                const data = await res.json();
                
                if (res.ok && data.access_token) {
                    const token = data.access_token;
                    localStorage.setItem('mock_token', token);
                    localStorage.setItem('mock_user', JSON.stringify(data.user));
                    this.user = data.user;
                    this.session = { access_token: token, user: data.user };
                    this._notifyListeners('SIGNED_IN', this.session);
                    return { data: { user: this.user, session: this.session }, error: null };
                } else {
                    return { data: null, error: { message: data.error || 'Signup failed' } };
                }
            } catch (e) {
                return { data: null, error: { message: 'Connection error' } };
            }
        }
        const { data, error } = await this.supabase.auth.signUp({
            email,
            password
        });
        return { data, error };
    }

    /**
     * Sign In with Email and Password
     */
    async signIn(email, password) {
        if (this.mockMode) {
            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password})
                });
                const data = await res.json();
                
                if (res.ok && data.access_token) {
                    const token = data.access_token;
                    localStorage.setItem('mock_token', token);
                    localStorage.setItem('mock_user', JSON.stringify(data.user));
                    this.user = data.user;
                    this.session = { access_token: token, user: data.user };
                    this._notifyListeners('SIGNED_IN', this.session);
                    return { data: { user: this.user, session: this.session }, error: null };
                } else {
                    return { data: null, error: { message: data.error || 'Login failed' } };
                }
            } catch (e) {
                return { data: null, error: { message: 'Connection error' } };
            }
        }

        const { data, error } = await this.supabase.auth.signInWithPassword({
            email,
            password
        });
        return { data, error };
    }

    /**
     * Sign Out
     */
    async signOut() {
        if (this.mockMode) {
            localStorage.removeItem('mock_token');
            this.user = null;
            this.session = null;
            this._notifyListeners('SIGNED_OUT', null);
            return { error: null };
        }
        const { error } = await this.supabase.auth.signOut();
        return { error };
    }

    /**
     * Get current session
     */
    async getSession() {
        if (this.mockMode) {
            return { data: { session: this.session }, error: null };
        }
        return await this.supabase.auth.getSession();
    }

    /**
     * Get default Authorization header value
     */
    getAuthHeader() {
        if (this.session && this.session.access_token) {
            return `Bearer ${this.session.access_token}`;
        }
        return null;
    }

    /**
     * Send password reset email
     */
    async resetPassword(email, redirectTo) {
        if (this.mockMode) {
             try {
                 const res = await fetch('/api/auth/reset-password', {
                     method: 'POST',
                     headers: {'Content-Type': 'application/json'},
                     body: JSON.stringify({email})
                 });
                 if (res.ok) {
                     return { data: {}, error: null };
                 } else {
                     return { data: null, error: { message: 'User not found' } };
                 }
             } catch(e) {
                 return { data: null, error: { message: 'Connection error' } };
             }
        }
        const { data, error } = await this.supabase.auth.resetPasswordForEmail(email, {
            redirectTo: redirectTo || window.location.origin
        });
        return { data, error };
    }

    /**
     * Update user (used for password update after reset)
     */
    async updatePassword(newPassword) {
        if (this.mockMode) {
            console.log('[MOCK] Password updated');
            return { data: { user: this.user }, error: null };
        }
        const { data, error } = await this.supabase.auth.updateUser({
            password: newPassword
        });
        return { data, error };
    }
}
