// static/js/dashboard.js
$(document).ready(function() {
    // Initial load of jobs
    loadJobs();
    
    // Set up automatic refresh every 60 seconds
    setInterval(loadJobs, 60000);
    
    // Set up tab switching
    $('#dashboard-tabs a').on('click', function(e) {
        e.preventDefault();
        $(this).tab('show');
    });
});

function loadJobs() {
    $.ajax({
        url: '/get_offline_jobs',
        type: 'GET',
        success: function(jobs) {
            updateJobTables(jobs);
        },
        error: function(xhr, status, error) {
            console.error('Error loading jobs:', error);
        }
    });
}

function updateJobTables(jobs) {
    // Clear existing jobs
    $('#queued-jobs').empty();
    $('#processing-jobs').empty();
    $('#completed-jobs').empty();
    $('#failed-jobs').empty();
    
    // Group jobs by status
    const queuedJobs = jobs.filter(job => job.status === 'queued');
    const processingJobs = jobs.filter(job => job.status === 'processing');
    const completedJobs = jobs.filter(job => job.status === 'completed');
    const failedJobs = jobs.filter(job => job.status === 'failed');
    
    // Update count badges
    $('#queued-count').text(queuedJobs.length);
    $('#processing-count').text(processingJobs.length);
    $('#completed-count').text(completedJobs.length);
    $('#failed-count').text(failedJobs.length);
    
    // Add rows to each table
    queuedJobs.forEach(job => addJobRow('queued-jobs', job));
    processingJobs.forEach(job => addJobRow('processing-jobs', job));
    completedJobs.forEach(job => addJobRow('completed-jobs', job));
    failedJobs.forEach(job => addJobRow('failed-jobs', job));
    
    // Show empty message if no jobs
    if (queuedJobs.length === 0) {
        $('#queued-jobs').html('<tr><td colspan="5" class="text-center">No queued jobs</td></tr>');
    }
    if (processingJobs.length === 0) {
        $('#processing-jobs').html('<tr><td colspan="5" class="text-center">No jobs in progress</td></tr>');
    }
    if (completedJobs.length === 0) {
        $('#completed-jobs').html('<tr><td colspan="5" class="text-center">No completed jobs</td></tr>');
    }
    if (failedJobs.length === 0) {
        $('#failed-jobs').html('<tr><td colspan="5" class="text-center">No failed jobs</td></tr>');
    }
}

function addJobRow(tableId, job) {
    // Format timestamps for display
    const queuedAt = new Date(job.queued_at || job.timestamp).toLocaleString();
    const startedAt = job.processing_started ? new Date(job.processing_started).toLocaleString() : 'N/A';
    const completedAt = job.completed_at ? new Date(job.completed_at).toLocaleString() : 'N/A';
    
    // Determine status class
    let statusClass = 'bg-secondary';
    if (job.status === 'queued') statusClass = 'bg-warning text-dark';
    if (job.status === 'processing') statusClass = 'bg-primary';
    if (job.status === 'completed') statusClass = 'bg-success';
    if (job.status === 'failed') statusClass = 'bg-danger';
    
    // Create actions buttons based on status
    let actionsHtml = '';
    if (job.status === 'completed') {
        // Link to view results 
        actionsHtml = `
            <a href="/review/session/${job.session_id}" class="btn btn-sm btn-primary">View Results</a>
            <button onclick="deleteJob('${job.session_id}')" class="btn btn-sm btn-danger">Delete</button>
        `;
    } else if (job.status === 'failed') {
        // Show error and delete button
        actionsHtml = `
            <button onclick="showError('${job.session_id}')" class="btn btn-sm btn-warning">Show Error</button>
            <button onclick="deleteJob('${job.session_id}')" class="btn btn-sm btn-danger">Delete</button>
        `;
    } else {
        // For queued and processing, just show cancel button
        actionsHtml = `
            <button onclick="deleteJob('${job.session_id}')" class="btn btn-sm btn-danger">Cancel</button>
        `;
    }
    
    // Create the row HTML
    const html = `
        <tr data-job-id="${job.session_id}">
            <td>${job.session_id}</td>
            <td><span class="badge ${statusClass}">${job.status}</span></td>
            <td>${queuedAt}</td>
            <td>${job.status === 'queued' ? 'Waiting' : (job.status === 'processing' ? startedAt : completedAt)}</td>
            <td>${actionsHtml}</td>
        </tr>
    `;
    
    // Add to the table
    $(`#${tableId}`).append(html);
}

function showError(jobId) {
    // Fetch job details to get the error message
    $.ajax({
        url: `/get_job_status/${jobId}`,
        type: 'GET',
        success: function(job) {
            if (job.error) {
                alert(`Error processing job: ${job.error}`);
            } else {
                alert('No error information available');
            }
        },
        error: function() {
            alert('Failed to fetch job details');
        }
    });
}

function deleteJob(jobId) {
    if (confirm('Are you sure you want to delete this job? This action cannot be undone.')) {
        $.ajax({
            url: `/api/offline_jobs/${jobId}`,
            type: 'DELETE',
            success: function() {
                // Reload jobs after deletion
                loadJobs();
            },
            error: function() {
                alert('Failed to delete job');
            }
        });
    }
}