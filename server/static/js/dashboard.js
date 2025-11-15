// State Management
        let currentPersonId = null;
        let currentPersonData = null;
        let streamInterval = null;
        let calibrationInterval = null;
        let calibrationCanvas = null;
        let calibrationCtx = null;
        let isDragging = false;
        let dragLine = null;
        let outerX = 800;
        let innerX = 1800;
        const RPI_ID = 'default';
        let currentTab = 'unlabeled';
        let autoRefresh = true;
        let autoRefreshInterval = null;
        let allPeopleData = [];
        let activityHistory = [];
        let lastConnectionCheck = Date.now();
        let isConnected = true;

        // Theme Management
        function toggleTheme() {
            document.body.classList.toggle('light-mode');
            const isLight = document.body.classList.contains('light-mode');
            document.getElementById('themeIcon').textContent = isLight ? '🌙' : '☀️';
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
        }

        // Load saved theme
        if (localStorage.getItem('theme') === 'light') {
            document.body.classList.add('light-mode');
            document.getElementById('themeIcon').textContent = '🌙';
        }

        // Auto-refresh Toggle
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const switchEl = document.getElementById('autoRefreshSwitch');
            
            if (autoRefresh) {
                switchEl.classList.add('active');
                startAutoRefresh();
            } else {
                switchEl.classList.remove('active');
                stopAutoRefresh();
            }
        }

        function startAutoRefresh() {
            if (autoRefreshInterval) clearInterval(autoRefreshInterval);
            // Refresh every 10 seconds (reduced frequency to prevent overload)
            autoRefreshInterval = setInterval(refresh, 10000);
        }

        function stopAutoRefresh() {
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
            }
        }

        // Connection Status Monitor
        function updateServerConnectionStatus(connected) {
            isConnected = connected;
            const dot = document.getElementById('serverConnectionDot');
            const text = document.getElementById('serverConnectionText');
            
            if (connected) {
                dot.classList.remove('disconnected');
                text.textContent = 'Server: Connected';
                // Re-enable auto-refresh if it was disabled
                if (!autoRefresh && autoRefreshInterval === null) {
                    // Don't auto-enable, let user control it
                }
            } else {
                dot.classList.add('disconnected');
                text.textContent = 'Server: Disconnected';
                // Don't turn off auto-refresh - keep trying to reconnect
            }
        }

        function updateRPiConnectionStatus(connected, isStreaming = false) {
            const dot = document.getElementById('rpiConnectionDot');
            const text = document.getElementById('rpiConnectionText');
            
            if (connected) {
                dot.classList.remove('disconnected');
                text.textContent = isStreaming ? 'RPi: Streaming' : 'RPi: Connected';
            } else {
                dot.classList.add('disconnected');
                text.textContent = 'RPi: Disconnected';
            }
        }

        // Toast Notifications
        function showToast(message, type = 'info', duration = 3000) {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            
            toast.innerHTML = `
                <div class="toast-content">${message}</div>
                <span class="toast-close" onclick="this.parentElement.remove()">×</span>
            `;
            
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 300);
            }, duration);
        }

        // Tab Switching
        function switchTab(tab) {
            currentTab = tab;
            
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
            document.getElementById(tab + 'Tab').classList.add('active');
            
            if (tab === 'unlabeled') {
                fetchUnlabeled();
            } else if (tab === 'people') {
                filterPeople(); // Use filterPeople which will show data if available
            }
        }

        // Track consecutive failures for more resilient connection status
        let connectionFailureCount = 0;
        const MAX_FAILURES = 3; // Allow 3 failures before marking as disconnected
        
        // Check RPi Status with timeout
        async function checkRPiStatus() {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout (increased from 2)
                
                const response = await fetch('/api/rpi/status', {
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                
                // Success - reset failure count and update status
                connectionFailureCount = 0;
                updateServerConnectionStatus(true);
                lastConnectionCheck = Date.now();
                
                if (data.rpis && data.rpis.length > 0) {
                    const rpi = data.rpis[0];
                    // Check if RPi is actually active (last_seen within last 60 seconds)
                    const lastSeen = new Date(rpi.last_seen).getTime();
                    const now = Date.now();
                    const secondsSinceLastSeen = (now - lastSeen) / 1000;
                    const isActive = secondsSinceLastSeen < 60; // 60 second timeout
                    
                    if (isActive) {
                        const isStreaming = rpi.is_streaming;
                        updateRPiConnectionStatus(true, isStreaming);
                    } else {
                        updateRPiConnectionStatus(false);
                    }
                } else {
                    updateRPiConnectionStatus(false);
                }
            } catch (error) {
                connectionFailureCount++;
                
                // Only mark as disconnected after multiple consecutive failures
                if (connectionFailureCount >= MAX_FAILURES) {
                    if (error.name !== 'AbortError') {
                        console.error('RPi status check error:', error);
                    }
                    updateServerConnectionStatus(false);
                    updateRPiConnectionStatus(false);
                }
                // Otherwise, keep current status (don't change on single failure)
            }
        }

        // Show RPi Status Modal
        async function showRPiStatus() {
            try {
                const response = await fetch('/api/rpi/status');
                const data = await response.json();
                
                let html = '<div style="font-size: 13px;">';
                if (data.rpis && data.rpis.length > 0) {
                    data.rpis.forEach(rpi => {
                        html += `
                            <div style="background: var(--bg-tertiary); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                                <div><strong>RPi ID:</strong> ${rpi.rpi_id}</div>
                                <div><strong>Status:</strong> ${rpi.status}</div>
                                <div><strong>Streaming:</strong> ${rpi.is_streaming ? 'Yes' : 'No'}</div>
                                <div><strong>Uptime:</strong> ${Math.floor(rpi.uptime / 60)}m ${rpi.uptime % 60}s</div>
                                <div><strong>Last Seen:</strong> ${new Date(rpi.last_seen).toLocaleTimeString()}</div>
                            </div>
                        `;
                    });
                } else {
                    html += '<div class="empty-state"><div class="emoji">🔌</div><h3>No RPi devices connected</h3></div>';
                }
                html += '</div>';
                
                document.getElementById('rpiStatusDetails').innerHTML = html;
                document.getElementById('rpiStatusModal').classList.add('active');
            } catch (error) {
                showToast('Failed to fetch RPi status', 'error');
            }
        }

        function closeRPiStatusModal() {
            document.getElementById('rpiStatusModal').classList.remove('active');
        }

        // Fetch Stats with timeout
        async function fetchStats() {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout (increased from 3)
                
                const response = await fetch('/api/dashboard/stats', {
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                
                document.getElementById('presentCount').textContent = data.people_present;
                document.getElementById('totalCount').textContent = data.total_people;
                document.getElementById('unlabeledCount').textContent = data.unlabeled_count;
                document.getElementById('vectorCount').textContent = data.vector_count;
                
                // Stats fetch success indicates server connection
                updateServerConnectionStatus(true);
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Fetch stats error:', error);
                }
                // Don't mark as disconnected on single failure - let checkRPiStatus handle it
            }
        }

        // Activity Feed Management
        function updateActivityFeed(people) {
            people.forEach(person => {
                const existing = activityHistory.find(a => a.person_id === person.person_id);
                
                // Use actual event timestamp (entered_at for entry, last_exit for exit, or last_seen as fallback)
                let eventTimestamp;
                if (person.state === 'in' && person.entered_at) {
                    eventTimestamp = new Date(person.entered_at).getTime();
                } else if (person.state === 'out' && person.last_exit) {
                    eventTimestamp = new Date(person.last_exit).getTime();
                } else {
                    eventTimestamp = new Date(person.last_seen).getTime();
                }
                
                if (!existing) {
                    activityHistory.unshift({
                        person_id: person.person_id,
                        name: person.name,
                        state: person.state,
                        last_seen: person.last_seen,
                        timestamp: eventTimestamp, // Use actual event time
                        action: person.state === 'in' ? 'enter' : 'leave',
                        is_labeled: person.is_labeled
                    });
                } else if (existing.state !== person.state) {
                    // State changed - update existing entry with new timestamp instead of creating new one
                    existing.state = person.state;
                    existing.timestamp = eventTimestamp; // Update timestamp to actual event time
                    existing.action = person.state === 'in' ? 'enter' : 'leave';
                    existing.last_seen = person.last_seen;
                    // Move to top of list
                    activityHistory = activityHistory.filter(a => a.person_id !== person.person_id);
                    activityHistory.unshift(existing);
                } else {
                    // State unchanged - just update last_seen, don't change timestamp
                    existing.last_seen = person.last_seen;
                }
            });
            
            activityHistory = activityHistory.slice(0, 20);
            
            // Render activity feed with smooth updates
            renderActivityListSmooth(); 
        }

        function renderActivityList() {
            const activityList = document.getElementById('activityList');
            activityList.innerHTML = '';
            
            if (activityHistory.length === 0) {
                activityList.innerHTML = '<div class="empty-state"><p>No recent activity</p></div>';
                return;
            }
            
            activityHistory.forEach(activity => {
                const item = document.createElement('div');
                item.className = 'activity-item';
                item.onclick = () => {
                    const person = allPeopleData.find(p => p.person_id === activity.person_id);
                    if (person) openPersonModal(person);
                };
                
                const timeAgo = getTimeAgo(activity.timestamp);
                const iconClass = activity.is_labeled ? activity.action : 'unlabeled';
                const actionText = activity.action === 'enter' ? 'entered' : 'left';
                
                item.innerHTML = `
                    <div class="activity-icon ${iconClass}"></div>
                    <div class="activity-text">
                        <div class="activity-name ${activity.is_labeled ? '' : 'unlabeled'}">
                            ${activity.name || 'Unlabeled'}
                        </div>
                        <div>${actionText}</div>
                    </div>
                    <div class="activity-time">${timeAgo}</div>

                    ${!activity.is_labeled ? 
                        `<span class="activity-action" onclick="event.stopPropagation(); openPersonModal(allPeopleData.find(p => p.person_id === '${activity.person_id}'))">Label</span>` 
                        : ''}

                    <span class="activity-delete" data-timestamp="${activity.timestamp}">×</span>
                `;
                
                // Add event listener for delete button
                const deleteBtn = item.querySelector('.activity-delete');
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    deleteActivity(activity.timestamp, e);
                });
                
                activityList.appendChild(item);
            });
        }

        // Smooth update version - updates in place without clearing
        function renderActivityListSmooth() {
            const activityList = document.getElementById('activityList');
            
            if (activityHistory.length === 0) {
                activityList.innerHTML = '<div class="empty-state"><p>No recent activity</p></div>';
                return;
            }
            
            // Get existing items
            const existingItems = Array.from(activityList.children);
            const existingMap = new Map();
            existingItems.forEach(item => {
                const deleteBtn = item.querySelector('.activity-delete');
                if (deleteBtn) {
                    const timestamp = parseInt(deleteBtn.getAttribute('data-timestamp'));
                    existingMap.set(timestamp, item);
                }
            });
            
            // Create a map of current activities
            const currentMap = new Map();
            activityHistory.forEach(activity => {
                currentMap.set(activity.timestamp, activity);
            });
            
            // Remove items that no longer exist
            existingMap.forEach((item, timestamp) => {
                if (!currentMap.has(timestamp)) {
                    item.remove();
                }
            });
            
            // Update or create items
            activityHistory.forEach((activity, index) => {
                const existingItem = existingMap.get(activity.timestamp);
                
                if (existingItem) {
                    // Update existing item in place
                    const timeElement = existingItem.querySelector('.activity-time');
                    if (timeElement) {
                        timeElement.textContent = getTimeAgo(activity.timestamp);
                    }
                    // Update name if changed
                    const nameElement = existingItem.querySelector('.activity-name');
                    if (nameElement) {
                        nameElement.textContent = activity.name || 'Unlabeled';
                    }
                    // Move to correct position if needed
                    if (existingItem.nextSibling !== activityList.children[index]) {
                        activityList.insertBefore(existingItem, activityList.children[index]);
                    }
                } else {
                    // Create new item
                    const item = document.createElement('div');
                    item.className = 'activity-item';
                    item.onclick = () => {
                        const person = allPeopleData.find(p => p.person_id === activity.person_id);
                        if (person) openPersonModal(person);
                    };
                    
                    const timeAgo = getTimeAgo(activity.timestamp);
                    const iconClass = activity.is_labeled ? activity.action : 'unlabeled';
                    const actionText = activity.action === 'enter' ? 'entered' : 'left';
                    
                    item.innerHTML = `
                        <div class="activity-icon ${iconClass}"></div>
                        <div class="activity-text">
                            <div class="activity-name ${activity.is_labeled ? '' : 'unlabeled'}">
                                ${activity.name || 'Unlabeled'}
                            </div>
                            <div>${actionText}</div>
                        </div>
                        <div class="activity-time">${timeAgo}</div>

                        ${!activity.is_labeled ? 
                            `<span class="activity-action" onclick="event.stopPropagation(); openPersonModal(allPeopleData.find(p => p.person_id === '${activity.person_id}'))">Label</span>` 
                            : ''}

                        <span class="activity-delete" data-timestamp="${activity.timestamp}">×</span>
                    `;
                    
                    const deleteBtn = item.querySelector('.activity-delete');
                    deleteBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        deleteActivity(activity.timestamp, e);
                    });
                    
                    // Insert at correct position
                    if (index < activityList.children.length) {
                        activityList.insertBefore(item, activityList.children[index]);
                    } else {
                        activityList.appendChild(item);
                    }
                }
            });
        }

        // Who's Here Now
        function updateWhosHere(people) {
            const presentPeople = people.filter(p => p.state === 'in');
            const presentList = document.getElementById('presentList');
            const presentCountInline = document.getElementById('presentCountInline');
            
            presentCountInline.textContent = presentPeople.length;
            presentList.innerHTML = '';
            
            if (presentPeople.length === 0) {
                presentList.innerHTML = '<div class="empty-state"><p>No one here yet</p></div>';
                return;
            }
            
            presentPeople.forEach(person => {
                const div = document.createElement('div');
                div.className = 'present-person';
                div.onclick = () => openPersonModal(person);
                
                // Use entered_at if available (when they actually entered), otherwise fall back to last_seen
                const entryTime = person.entered_at ? new Date(person.entered_at).getTime() : new Date(person.last_seen).getTime();
                const enteredAgo = getTimeAgo(entryTime);
                
                div.innerHTML = `
                    <div class="present-avatar">
                        ${person.thumbnail ? `<img src="${person.thumbnail}">` : ''}
                    </div>
                    <div class="present-info">
                        <div class="present-name">${person.name || 'Unlabeled'}</div>
                        <div class="present-time">Entered ${enteredAgo}</div>
                    </div>
                `;
                
                presentList.appendChild(div);
            });
        }

        // Today's Summary (Mock data for now)
        function updateTodaySummary(people) {
            // This would ideally come from a dedicated endpoint with event history
            const totalEntries = activityHistory.filter(a => a.action === 'enter').length;
            const uniqueVisitors = new Set(activityHistory.map(a => a.person_id)).size;
            const unlabeled = people.filter(p => !p.is_labeled).length;
            
            document.getElementById('todayEntries').textContent = totalEntries;
            document.getElementById('todayUnique').textContent = uniqueVisitors;
            document.getElementById('todayUnlabeled').textContent = unlabeled;
            document.getElementById('peakTime').textContent = 'N/A';
        }

        // Time ago helper
        function getTimeAgo(timestamp) {
            // Handle both ISO string and timestamp number
            const time = typeof timestamp === 'string' ? new Date(timestamp).getTime() : timestamp;
            const now = Date.now();
            const seconds = Math.floor((now - time) / 1000);
            
            // If timestamp is in the future or very recent (within 5 seconds), show "just now"
            if (seconds < 0 || seconds < 5) return 'just now';
            if (seconds < 60) return `${seconds}s ago`;
            if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
            if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
            return `${Math.floor(seconds / 86400)}d ago`;
        }

        function deleteActivity(timestamp, event) {
            event.stopPropagation();
            
            activityHistory = activityHistory.filter(item => item.timestamp !== timestamp);
            
            renderActivityListSmooth();
        }

        // Fetch Unlabeled
        async function fetchUnlabeled() {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
                
                const response = await fetch('/api/dashboard/people', {
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const people = await response.json();
                const unlabeled = people.filter(p => !p.is_labeled);
                
                const grid = document.getElementById('unlabeledGrid');
                
                // Remove any skeleton cards immediately
                const skeletons = grid.querySelectorAll('.skeleton-card');
                skeletons.forEach(s => s.remove());
                
                // If no unlabeled people, show empty state immediately
                if (unlabeled.length === 0) {
                    grid.innerHTML = `
                        <div class="empty-state">
                            <h3>No unlabeled people</h3>
                        </div>
                    `;
                    return;
                }
                
                // Smooth update - use data attributes to track person_id
                const existingCards = Array.from(grid.children).filter(card => !card.classList.contains('empty-state'));
                const existingIds = new Set();
                existingCards.forEach(card => {
                    const personId = card.getAttribute('data-person-id');
                    if (personId) {
                        existingIds.add(personId);
                    }
                });
                
                const unlabeledIds = new Set(unlabeled.map(p => p.person_id));
                
                // Remove cards that are no longer unlabeled
                existingCards.forEach(card => {
                    const personId = card.getAttribute('data-person-id');
                    if (personId && !unlabeledIds.has(personId)) {
                        card.remove();
                    }
                });
                
                // Remove empty state if we have data
                const existingEmpty = grid.querySelector('.empty-state');
                if (existingEmpty) existingEmpty.remove();
                
                // Add new unlabeled cards
                unlabeled.forEach(person => {
                    const existingCard = grid.querySelector(`[data-person-id="${person.person_id}"]`);
                    
                    if (!existingCard) {
                        const newCard = createPersonCard(person, true);
                        newCard.setAttribute('data-person-id', person.person_id);
                        grid.appendChild(newCard);
                    }
                });
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Fetch unlabeled error:', error);
                    showToast('Failed to fetch unlabeled people', 'error');
                }
            }
        }

        // Fetch All People with filtering
        async function fetchAllPeople() {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
                
                const response = await fetch('/api/dashboard/people', {
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                allPeopleData = await response.json();
                
                // Remove any skeleton cards immediately
                const grid = document.getElementById('allPeopleGrid');
                const skeletons = grid.querySelectorAll('.skeleton-card');
                skeletons.forEach(s => s.remove());
                
                filterPeople();
                
                // Update activity feed and who's here
                updateActivityFeed(allPeopleData);
                updateWhosHere(allPeopleData);
                updateTodaySummary(allPeopleData);
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Fetch people error:', error);
                    showToast('Failed to fetch people', 'error');
                }
            }
        }

        // Filter People (Search + Filter) - Smooth update version
        function filterPeople() {
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const filterType = document.getElementById('filterSelect').value;
            
            let filtered = allPeopleData;
            
            // Apply filter
            if (filterType === 'present') {
                filtered = filtered.filter(p => p.state === 'in');
            } else if (filterType === 'unlabeled') {
                filtered = filtered.filter(p => !p.is_labeled);
            }
            
            // Apply search
            if (searchTerm) {
                filtered = filtered.filter(p => 
                    (p.name && p.name.toLowerCase().includes(searchTerm)) ||
                    p.person_id.toLowerCase().includes(searchTerm)
                );
            }
            
            const grid = document.getElementById('allPeopleGrid');
            
            // Remove any skeleton cards immediately
            const skeletons = grid.querySelectorAll('.skeleton-card');
            skeletons.forEach(s => s.remove());
            
            // If no filtered results, show empty state immediately
            if (filtered.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state">
                        <h3>No people found</h3>
                        <p>Try adjusting your search or filter</p>
                    </div>
                `;
                return;
            }
            
            // Smooth update - use data attributes to track person_id
            const existingCards = Array.from(grid.children).filter(card => !card.classList.contains('empty-state'));
            const existingIds = new Set();
            existingCards.forEach(card => {
                const personId = card.getAttribute('data-person-id');
                if (personId) {
                    existingIds.add(personId);
                }
            });
            
            const filteredIds = new Set(filtered.map(p => p.person_id));
            
            // Remove cards that are no longer in filtered list
            existingCards.forEach(card => {
                const personId = card.getAttribute('data-person-id');
                if (personId && !filteredIds.has(personId)) {
                    card.remove();
                }
            });
            
            // Remove empty state if we have data
            const existingEmpty = grid.querySelector('.empty-state');
            if (existingEmpty) existingEmpty.remove();
            
            // Add new cards or update existing ones
            filtered.forEach(person => {
                const existingCard = grid.querySelector(`[data-person-id="${person.person_id}"]`);
                
                if (!existingCard) {
                    const newCard = createPersonCard(person, !person.is_labeled);
                    newCard.setAttribute('data-person-id', person.person_id);
                    grid.appendChild(newCard);
                }
            });
        }

        // Create Person Card
        function createPersonCard(person, unlabeled) {
            const card = document.createElement('div');
            card.className = 'person-card' + (unlabeled ? ' unlabeled' : '');
            card.onclick = () => {
                openPersonModal(person);
            };
            
            card.innerHTML = `
                <div class="card-image">
                    ${person.thumbnail ? 
                        `<img src="${person.thumbnail}">` :
                        `<div class="card-image-placeholder"></div>`
                    }
                    <div class="image-count-badge">${person.image_count}</div>
                </div>
                <div class="card-name ${unlabeled ? 'unlabeled' : ''}">${person.name || 'Click to label'}</div>
                ${person.name ? `<div class="card-id">${person.person_id}</div>` : ''}
            `;
            
            return card;
        }

        // Show loading skeletons
        function showLoadingSkeletons(gridId, count = 8) {
            const grid = document.getElementById(gridId);
            grid.innerHTML = '';
            
            for (let i = 0; i < count; i++) {
                const skeleton = document.createElement('div');
                skeleton.className = 'skeleton-card';
                skeleton.innerHTML = `
                    <div class="skeleton skeleton-image"></div>
                    <div class="skeleton skeleton-text" style="width: 80%;"></div>
                    <div class="skeleton skeleton-text" style="width: 60%;"></div>
                `;
                grid.appendChild(skeleton);
            }
        }

        // Refresh
        async function refresh() {
            const indicator = document.getElementById('loadingIndicator');
            if (indicator) {
                indicator.style.display = 'inline-block';
            }
            
            // Don't show loading skeletons - just update data smoothly
            // Empty states will show automatically if there's no data
            
            try {
                // Use Promise.allSettled so one failure doesn't block others
                const results = await Promise.allSettled([
                    fetchStats(),
                    checkRPiStatus(),
                    fetchAllPeople()
                ]);
                
                // Log any failures
                results.forEach((result, index) => {
                    if (result.status === 'rejected') {
                        const names = ['fetchStats', 'checkRPiStatus', 'fetchAllPeople'];
                        console.error(`${names[index]} failed:`, result.reason);
                    }
                });
                
                // Refresh current tab (only if fetchAllPeople succeeded)
                if (results[2].status === 'fulfilled') {
                    if (currentTab === 'unlabeled') {
                        await fetchUnlabeled();
                    } else if (currentTab === 'people') {
                        filterPeople();
                    }
                }
            } catch (error) {
                console.error('Refresh error:', error);
            } finally {
                if (indicator) {
                    indicator.style.display = 'none';
                }
            }
        }

        // Person Detail Modal
        function openPersonModal(person) {
            currentPersonData = person;
            currentPersonId = person.person_id;
            
            document.getElementById('detailId').textContent = person.name || person.person_id;
            document.getElementById('detailNameInput').value = person.name || '';
            document.getElementById('detailState').textContent = person.state;
            
            // Parse ISO timestamp and display in local timezone (ensure correct timezone handling)
            // ISO strings with 'Z' or timezone info are parsed correctly by JavaScript
            const options = { 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric', 
                hour: '2-digit', 
                minute: '2-digit',
                hour12: true,
                timeZoneName: 'short'
            };
            
            // Show last similarity (recognition confidence)
            const similarityField = document.getElementById('detailSimilarityField');
            if (person.last_similarity !== null && person.last_similarity !== undefined) {
                const similarityPercent = (person.last_similarity * 100).toFixed(1);
                document.getElementById('detailSimilarity').textContent = `${similarityPercent}%`;
                similarityField.style.display = 'block';
            } else {
                similarityField.style.display = 'none';
            }
            
            // Show entered_at and time in office if person is currently in
            const enteredAtField = document.getElementById('detailEnteredAtField');
            const timeInOfficeField = document.getElementById('detailTimeInOfficeField');
            if (person.state === 'in' && person.entered_at) {
                const enteredDate = new Date(person.entered_at);
                document.getElementById('detailEnteredAt').textContent = enteredDate.toLocaleString(undefined, options);
                enteredAtField.style.display = 'block';
                
                // Calculate time in office
                const now = Date.now();
                const enteredTime = enteredDate.getTime();
                const minutesInOffice = Math.floor((now - enteredTime) / 60000);
                const hours = Math.floor(minutesInOffice / 60);
                const minutes = minutesInOffice % 60;
                const timeInOfficeText = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
                document.getElementById('detailTimeInOffice').textContent = timeInOfficeText;
                timeInOfficeField.style.display = 'block';
            } else {
                enteredAtField.style.display = 'none';
                timeInOfficeField.style.display = 'none';
            }
            
            // Show last exit if person is out
            const lastExitField = document.getElementById('detailLastExitField');
            if (person.state === 'out' && person.last_exit) {
                const lastExitDate = new Date(person.last_exit);
                document.getElementById('detailLastExit').textContent = lastExitDate.toLocaleString(undefined, options);
                lastExitField.style.display = 'block';
            } else {
                lastExitField.style.display = 'none';
            }
            
            if (person.thumbnail) {
                document.getElementById('detailImage').src = person.thumbnail;
            } else {
                document.getElementById('detailImage').src = '';
            }
            
            document.getElementById('personDetailModal').classList.add('active');
            document.getElementById('detailNameInput').focus();
        }

        function closePersonModal() {
            document.getElementById('personDetailModal').classList.remove('active');
            currentPersonId = null;
            currentPersonData = null;
        }

        async function savePersonDetails() {
            const name = document.getElementById('detailNameInput').value.trim();
            
            if (!name) {
                showToast('Please enter a name', 'error');
                return;
            }
            
            try {
                await fetch('/api/dashboard/label', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        person_id: currentPersonId,
                        name: name
                    })
                });
                
                showToast(`${name} labeled successfully!`, 'success');
                closePersonModal();
                refresh();
            } catch (error) {
                showToast('Failed to label person', 'error');
            }
        }

        async function deletePersonFromModal() {
            if (!confirm(`Delete ${currentPersonData.name || currentPersonId}? This cannot be undone.`)) return;
            await deletePerson(currentPersonId);
            closePersonModal();
        }

        async function deletePerson(personId) {
            try {
                await fetch(`/api/dashboard/person/${personId}`, { method: 'DELETE' });
                showToast('Person deleted', 'success');
                refresh();
            } catch (error) {
                showToast('Failed to delete person', 'error');
            }
        }

        // Camera Functions
        async function viewCamera() {
            document.getElementById('cameraModal').classList.add('active');
            
            await fetch('/api/rpi/start-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rpi_id: RPI_ID })
            });
            
            setTimeout(() => {
                document.getElementById('streamPlaceholder').style.display = 'none';
                document.getElementById('streamImg').style.display = 'block';
            }, 1000);
            
            streamInterval = setInterval(() => {
                const img = document.getElementById('streamImg');
                img.src = `/api/rpi/stream/${RPI_ID}?t=${Date.now()}`;
            }, 200);
        }

        function closeCameraModal() {
            if (streamInterval) {
                clearInterval(streamInterval);
                streamInterval = null;
            }
            
            fetch('/api/rpi/stop-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rpi_id: RPI_ID })
            });
            
            document.getElementById('cameraModal').classList.remove('active');
            document.getElementById('streamImg').style.display = 'none';
            document.getElementById('streamPlaceholder').style.display = 'flex';
        }

        async function registerViaRPi() {
            if (!confirm('Trigger registration on RPi camera? This will capture 10 images automatically.')) {
                return;
            }
            
            try {
                const response = await fetch('/api/rpi/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rpi_id: RPI_ID })
                });
                
                const result = await response.json();
                showToast(result.message || 'Registration triggered on RPi', 'success');
                setTimeout(refresh, 5000);
            } catch (error) {
                showToast('Failed to trigger registration', 'error');
            }
        }

        function registerFromStream() {
            closeCameraModal();
            registerViaRPi();
        }

        // Calibration
        async function calibrateRPi() {
            try {
                const response = await fetch(`/api/rpi/config/${RPI_ID}`);
                const config = await response.json();
                outerX = config.tripwires.outer_x;
                innerX = config.tripwires.inner_x;
                
                document.getElementById('outerXInput').value = outerX;
                document.getElementById('innerXInput').value = innerX;
                
                await fetch('/api/rpi/calibrate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rpi_id: RPI_ID })
                });
                
                document.getElementById('calibrationModal').classList.add('active');
                
                calibrationCanvas = document.getElementById('calibrationCanvas');
                calibrationCtx = calibrationCanvas.getContext('2d');
                
                calibrationCanvas.addEventListener('mousemove', showMousePosition);
                
                calibrationInterval = setInterval(() => {
                    const img = new Image();
                    img.onload = () => {
                        calibrationCanvas.width = img.width;
                        calibrationCanvas.height = img.height;
                        calibrationCtx.drawImage(img, 0, 0);
                        drawCalibrationLines();
                    };
                    img.src = `/api/rpi/stream/${RPI_ID}?t=${Date.now()}`;
                }, 200);
                
                calibrationCanvas.addEventListener('mousedown', startDrag);
                calibrationCanvas.addEventListener('mousemove', drag);
                calibrationCanvas.addEventListener('mouseup', endDrag);
                calibrationCanvas.addEventListener('mouseleave', endDrag);
            } catch (error) {
                showToast('Failed to start calibration', 'error');
            }
        }

        function drawCalibrationLines() {
            if (!calibrationCtx || !calibrationCanvas) return;
            
            calibrationCtx.strokeStyle = 'yellow';
            calibrationCtx.lineWidth = 3;
            calibrationCtx.beginPath();
            calibrationCtx.moveTo(outerX, 0);
            calibrationCtx.lineTo(outerX, calibrationCanvas.height);
            calibrationCtx.stroke();
            
            calibrationCtx.strokeStyle = 'red';
            calibrationCtx.lineWidth = 3;
            calibrationCtx.beginPath();
            calibrationCtx.moveTo(innerX, 0);
            calibrationCtx.lineTo(innerX, calibrationCanvas.height);
            calibrationCtx.stroke();
            
            calibrationCtx.fillStyle = 'yellow';
            calibrationCtx.font = 'bold 14px monospace';
            calibrationCtx.fillText(`${outerX}px`, outerX + 10, 30);
            
            calibrationCtx.fillStyle = 'red';
            calibrationCtx.fillText(`${innerX}px`, innerX + 10, 60);
        }

        function showMousePosition(e) {
            const rect = calibrationCanvas.getBoundingClientRect();
            const x = Math.round(e.clientX - rect.left);
            const mousePos = document.getElementById('mousePosition');
            mousePos.textContent = `X: ${x}px`;
            mousePos.style.display = 'block';
            // Position relative to canvas container, not window
            const container = calibrationCanvas.parentElement;
            const containerRect = container.getBoundingClientRect();
            mousePos.style.left = (e.clientX - containerRect.left + 15) + 'px';
            mousePos.style.top = (e.clientY - containerRect.top - 25) + 'px';
        }

        function updateCalibrationFromInput() {
            outerX = parseInt(document.getElementById('outerXInput').value);
            innerX = parseInt(document.getElementById('innerXInput').value);
        }

        function startDrag(e) {
            const rect = calibrationCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            
            if (Math.abs(x - outerX) < 20) {
                isDragging = true;
                dragLine = 'outer';
            } else if (Math.abs(x - innerX) < 20) {
                isDragging = true;
                dragLine = 'inner';
            }
        }

        function drag(e) {
            if (!isDragging) return;
            
            const rect = calibrationCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            
            if (dragLine === 'outer') {
                outerX = Math.max(0, Math.min(Math.round(x), calibrationCanvas.width));
                document.getElementById('outerXInput').value = outerX;
            } else if (dragLine === 'inner') {
                innerX = Math.max(0, Math.min(Math.round(x), calibrationCanvas.width));
                document.getElementById('innerXInput').value = innerX;
            }
        }

        function endDrag() {
            isDragging = false;
            dragLine = null;
        }

        async function saveCalibration() {
            outerX = parseInt(document.getElementById('outerXInput').value);
            innerX = parseInt(document.getElementById('innerXInput').value);
            
            try {
                await fetch(`/api/rpi/config/${RPI_ID}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        tripwires: { outer_x: outerX, inner_x: innerX }
                    })
                });
                
                showToast('Tripwire configuration saved!', 'success');
                closeCalibrationModal();
            } catch (error) {
                showToast('Failed to save calibration', 'error');
            }
        }

        function closeCalibrationModal() {
            if (calibrationInterval) {
                clearInterval(calibrationInterval);
                calibrationInterval = null;
            }
            
            const mousePos = document.getElementById('mousePosition');
            mousePos.style.display = 'none';
            
            fetch('/api/rpi/stop-calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rpi_id: RPI_ID })
            });
            
            document.getElementById('calibrationModal').classList.remove('active');
        }

        // Keyboard Shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closePersonModal();
                closeCameraModal();
                closeCalibrationModal();
                closeRPiStatusModal();
            }
            if (e.key === 'Enter' && document.getElementById('personDetailModal').classList.contains('active')) {
                savePersonDetails();
            }
            if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                document.getElementById('searchInput').focus();
            }
        });

        // Full State Management (Vectors + Metadata)
        async function downloadFullState() {
            try {
                const response = await fetch('/api/export/full-state');
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `full_state_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                showToast('Full state backup downloaded', 'success');
            } catch (error) {
                showToast('Failed to download full state backup', 'error');
            }
        }

        async function uploadFullState(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            if (!confirm('This will replace all current data (vectors and metadata). Are you sure?')) {
                event.target.value = '';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/import/full-state', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                if (response.ok) {
                    showToast(`Full state restored: ${result.people_imported} people, ${result.vectors_imported} vectors`, 'success');
                    // Reload the page to reflect changes
                    setTimeout(() => window.location.reload(), 1000);
                } else {
                    showToast('Failed to upload full state backup', 'error');
                }
            } catch (error) {
                showToast('Failed to upload full state backup', 'error');
            }
            
            // Reset file input
            event.target.value = '';
        }

        // Vector DB Management
        async function downloadVectorDB() {
            try {
                const response = await fetch('/api/export/vectors');
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `vectors_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                showToast('Vector database downloaded', 'success');
            } catch (error) {
                showToast('Failed to download vector database', 'error');
            }
        }

        // Similarity Matrix - Define directly on window to ensure availability
        window.showSimilarityMatrix = async function showSimilarityMatrix() {
            console.log('showSimilarityMatrix called');
            const modal = document.getElementById('similarityMatrixModal');
            const content = document.getElementById('similarityMatrixContent');
            
            if (!modal) {
                console.error('Similarity matrix modal not found');
                showToast('Error: Modal not found', 'error');
                return;
            }
            
            if (!content) {
                console.error('Similarity matrix content element not found');
                showToast('Error: Content element not found', 'error');
                return;
            }
            
            console.log('Showing modal');
            modal.classList.add('active');
            content.innerHTML = '<div style="text-align: center; padding: 20px;">Loading...</div>';
            
            try {
                console.log('Fetching similarity matrix...');
                const response = await fetch('/api/vector-similarity-matrix');
                console.log('Response status:', response.status);
                const data = await response.json();
                
                if (data.status === 'empty') {
                    content.innerHTML = `<div style="text-align: center; padding: 20px; color: var(--text-secondary);">${data.message}</div>`;
                    return;
                }
                
                if (data.status !== 'success') {
                    content.innerHTML = `<div style="text-align: center; padding: 20px; color: var(--text-danger);">Error loading similarity matrix</div>`;
                    return;
                }
                
                // Build HTML for similarity matrix
                let html = '<div style="margin-bottom: 20px;">';
                html += `<div style="margin-bottom: 10px;"><strong>Statistics:</strong></div>`;
                html += `<div style="margin-bottom: 5px;">Total Vectors: ${data.stats.count}</div>`;
                html += `<div style="margin-bottom: 5px;">Min Similarity: ${data.stats.min_similarity}</div>`;
                html += `<div style="margin-bottom: 5px;">Max Similarity: ${data.stats.max_similarity}</div>`;
                html += `<div style="margin-bottom: 15px;">Avg Similarity: ${data.stats.avg_similarity}</div>`;
                html += '</div>';
                
                // Person info table
                html += '<div style="margin-bottom: 20px; overflow-x: auto;">';
                html += '<table style="width: 100%; border-collapse: collapse; font-size: 12px;">';
                html += '<thead><tr style="background: var(--bg-secondary);">';
                html += '<th style="padding: 8px; text-align: left; border: 1px solid var(--border-color);">Person ID</th>';
                html += '<th style="padding: 8px; text-align: left; border: 1px solid var(--border-color);">Name</th>';
                html += '<th style="padding: 8px; text-align: left; border: 1px solid var(--border-color);">Vector Norm</th>';
                html += '</tr></thead><tbody>';
                
                data.person_info.forEach(info => {
                    html += '<tr>';
                    html += `<td style="padding: 8px; border: 1px solid var(--border-color);">${info.person_id}</td>`;
                    html += `<td style="padding: 8px; border: 1px solid var(--border-color);">${info.name || '(unlabeled)'}</td>`;
                    html += `<td style="padding: 8px; border: 1px solid var(--border-color);">${info.vector_norm}</td>`;
                    html += '</tr>';
                });
                
                html += '</tbody></table>';
                html += '</div>';
                
                // Similarity matrix table
                html += '<div style="overflow-x: auto; overflow-y: auto; max-height: 60vh;">';
                html += '<table style="width: 100%; border-collapse: collapse; font-size: 11px; font-family: monospace;">';
                
                // Header row
                html += '<thead><tr style="background: var(--bg-secondary); position: sticky; top: 0;">';
                html += '<th style="padding: 6px; text-align: left; border: 1px solid var(--border-color); min-width: 100px;">Person</th>';
                data.person_ids.forEach(personId => {
                    const info = data.person_info.find(p => p.person_id === personId);
                    const label = info?.name || personId;
                    html += `<th style="padding: 6px; text-align: center; border: 1px solid var(--border-color); min-width: 80px; writing-mode: vertical-rl; text-orientation: mixed;">${label}</th>`;
                });
                html += '</tr></thead><tbody>';
                
                // Data rows
                data.matrix.forEach((row, i) => {
                    const personId = data.person_ids[i];
                    const info = data.person_info.find(p => p.person_id === personId);
                    const label = info?.name || personId;
                    
                    html += '<tr>';
                    html += `<td style="padding: 6px; border: 1px solid var(--border-color); font-weight: 600; background: var(--bg-secondary);">${label}</td>`;
                    
                    row.forEach((similarity, j) => {
                        // Color code based on similarity
                        let bgColor = '';
                        let textColor = 'var(--text-primary)';
                        
                        if (i === j) {
                            // Diagonal (same person) = 1.0
                            bgColor = 'rgba(0, 200, 0, 0.2)';
                        } else if (similarity >= 0.9) {
                            // Very high similarity (suspicious)
                            bgColor = 'rgba(255, 0, 0, 0.3)';
                            textColor = '#ff0000';
                        } else if (similarity >= 0.7) {
                            // High similarity (concerning)
                            bgColor = 'rgba(255, 165, 0, 0.2)';
                            textColor = '#ff8800';
                        } else if (similarity >= 0.5) {
                            // Medium similarity
                            bgColor = 'rgba(255, 255, 0, 0.1)';
                        } else {
                            // Low similarity (good)
                            bgColor = 'rgba(0, 200, 0, 0.1)';
                        }
                        
                        html += `<td style="padding: 6px; text-align: center; border: 1px solid var(--border-color); background: ${bgColor}; color: ${textColor}; font-weight: ${similarity >= 0.7 ? '600' : '400'};">
                            ${similarity.toFixed(4)}
                        </td>`;
                    });
                    
                    html += '</tr>';
                });
                
                html += '</tbody></table>';
                html += '</div>';
                
                // Legend
                html += '<div style="margin-top: 20px; padding: 10px; background: var(--bg-secondary); border-radius: 4px; font-size: 12px;">';
                html += '<strong>Legend:</strong> ';
                html += '<span style="display: inline-block; width: 20px; height: 20px; background: rgba(0, 200, 0, 0.2); border: 1px solid var(--border-color); margin: 0 5px; vertical-align: middle;"></span> Same person (1.0) | ';
                html += '<span style="display: inline-block; width: 20px; height: 20px; background: rgba(0, 200, 0, 0.1); border: 1px solid var(--border-color); margin: 0 5px; vertical-align: middle;"></span> Low similarity (&lt;0.5, good) | ';
                html += '<span style="display: inline-block; width: 20px; height: 20px; background: rgba(255, 255, 0, 0.1); border: 1px solid var(--border-color); margin: 0 5px; vertical-align: middle;"></span> Medium (0.5-0.7) | ';
                html += '<span style="display: inline-block; width: 20px; height: 20px; background: rgba(255, 165, 0, 0.2); border: 1px solid var(--border-color); margin: 0 5px; vertical-align: middle;"></span> High (0.7-0.9, concerning) | ';
                html += '<span style="display: inline-block; width: 20px; height: 20px; background: rgba(255, 0, 0, 0.3); border: 1px solid var(--border-color); margin: 0 5px; vertical-align: middle;"></span> Very High (≥0.9, suspicious)';
                html += '</div>';
                
                content.innerHTML = html;
            } catch (error) {
                console.error('Error loading similarity matrix:', error);
                content.innerHTML = `<div style="text-align: center; padding: 20px; color: var(--text-danger);">Error loading similarity matrix: ${error.message}</div>`;
            }
        };

        window.closeSimilarityMatrix = function closeSimilarityMatrix() {
            document.getElementById('similarityMatrixModal').classList.remove('active');
        };

        async function uploadVectorDB(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/import/vectors', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                if (response.ok) {
                    showToast(`Vector database uploaded: ${result.vectors_imported} vectors`, 'success');
                    refresh();
                } else {
                    showToast('Failed to upload vector database', 'error');
                }
            } catch (error) {
                showToast('Failed to upload vector database', 'error');
            }
            
            // Reset file input
            event.target.value = '';
        }

        // System Toggle (Enable/Disable RPi)
        let systemEnabled = true;
        async function toggleSystem() {
            systemEnabled = !systemEnabled;
            const btn = document.getElementById('systemToggleBtn');
            
            try {
                const response = await fetch('/api/rpi/system-toggle', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        rpi_id: RPI_ID,
                        enabled: systemEnabled 
                    })
                });
                
                const result = await response.json();
                if (response.ok) {
                    btn.textContent = systemEnabled ? 'Disable System' : 'Enable System';
                    // Update button classes for color coding
                    btn.classList.remove('enabled', 'disabled');
                    btn.classList.add(systemEnabled ? 'enabled' : 'disabled');
                    showToast(result.message || (systemEnabled ? 'System enabled' : 'System disabled'), 'success');
                } else {
                    systemEnabled = !systemEnabled; // Revert on error
                    showToast('Failed to toggle system', 'error');
                }
            } catch (error) {
                systemEnabled = !systemEnabled; // Revert on error
                showToast('Failed to toggle system', 'error');
            }
        }

        // Password Protection
        function checkPassword() {
            const input = document.getElementById('passwordInput');
            const password = input.value;
            const errorDiv = document.getElementById('passwordError');
            
            // Check if last 3 characters are "0"
            if (password.length >= 3 && password.slice(-3) === '000') {
                // Valid password
                document.getElementById('passwordModal').classList.add('hidden');
                document.getElementById('mainContainer').style.display = 'block';
                // Initial load after password is correct - don't wait, render immediately
                startAutoRefresh(); // Start auto-refresh immediately
                // Load data asynchronously without blocking
                setTimeout(() => {
                    refresh().catch(err => console.error('Initial refresh failed:', err));
                }, 100); // Small delay to ensure DOM is ready
            } else {
                // Invalid password
                errorDiv.classList.add('show');
                input.value = '';
                input.focus();
            }
        }

        function handlePasswordKeyPress(event) {
            if (event.key === 'Enter') {
                checkPassword();
            }
        }

        // Focus password input on load
        window.addEventListener('load', () => {
            const passwordInput = document.getElementById('passwordInput');
            if (passwordInput) {
                passwordInput.focus();
            }
        });

        // Add event listener for similarity matrix button (backup to onclick)
        window.addEventListener('load', () => {
            const btn = document.getElementById('showSimilarityMatrixBtn');
            if (btn && !btn.onclick) {
                // Only add if onclick handler wasn't set in HTML
                btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    console.log('Similarity matrix button clicked (via event listener)');
                    if (typeof window.showSimilarityMatrix === 'function') {
                        window.showSimilarityMatrix();
                    } else {
                        console.error('showSimilarityMatrix function not found');
                        showToast('Error: Function not loaded. Please refresh the page.', 'error');
                    }
                });
            }
        });